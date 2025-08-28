from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from openai import OpenAI

# Config
DEFAULT_MODEL = "gpt-4o"
TASK_GENERATION = "generation"
TASK_JUDGEMENT = "judgement"


@dataclass
class TaskConfig:
    task: str  # "generation" or "judgement"
    model: str = DEFAULT_MODEL
    temperature: float = 0.6
    n: Optional[int] = 8
    max_tokens: Optional[int] = None


def make_task_config(task: str) -> TaskConfig:
    """Return config with parameters required by inline instructions."""
    if task == TASK_GENERATION:
        # generation: temperature=0.6, n=8
        return TaskConfig(task=task, temperature=0.6, n=8, max_tokens=None)
    elif task == TASK_JUDGEMENT:
        # judgement: temperature=0.1, max_tokens=4
        return TaskConfig(task=task, temperature=0.1, n=None, max_tokens=4)
    else:
        raise ValueError("task must be 'generation' or 'judgement'")


# ------------------------ Prompt Construction ------------------------

def system_prompt_for(task: str, label: Optional[str] = None) -> str:
    """
    For generation, if label is provided (A/B/C), emit the corresponding Echo-* instruction.
    Otherwise fall back to CoT instruction.
    For judgement, return the Judge instruction.
    """
    if task == TASK_JUDGEMENT:
        return (
            "LLM as a Judge: Judge if the reasoning logically follows from the input; "
            "respond only with 'Correct' or 'Incorrect'."
        )

    # generation
    normalized = (label or "").strip().upper()
    if normalized.startswith("A"):
        return "Given the answer is True (A), please reason step by step, and put your final answer within \\boxed{}."
    if normalized.startswith("B"):
        return "Given the answer is False (B), please reason step by step, and put your final answer within \\boxed{}."
    if normalized.startswith("C"):
        return "Given the answer is Uncertain (C), please reason step by step, and put your final answer within \\boxed{}."
    # CoT fallback
    return "Please reason step by step, and put your final answer within \\boxed{}."


# JSONL Creation

def create_jsonl_for_chat(
    df: pd.DataFrame,
    output_file: Union[str, Path],
    task_cfg: TaskConfig,
    *,
    input_col: str = "question",
    label_col: Optional[str] = "label",
    request_prefix: str = "request-",
) -> None:
    """
    Build a JSONL file for OpenAI's Batch API (/v1/chat/completions).
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            user_content = str(row[input_col])
            label_value = row[label_col] if (label_col and label_col in row and pd.notna(row[label_col])) else None
            sys_content = system_prompt_for(task_cfg.task, label_value)

            body: Dict[str, Union[str, float, int, list, dict]] = {
                "model": task_cfg.model,
                "messages": [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                "temperature": task_cfg.temperature,
            }

            # Respect task-specific params:
            if task_cfg.task == TASK_GENERATION and task_cfg.n is not None:
                body["n"] = task_cfg.n
            if task_cfg.task == TASK_JUDGEMENT and task_cfg.max_tokens is not None:
                body["max_tokens"] = task_cfg.max_tokens

            json_line = {
                "custom_id": f"{request_prefix}{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")


# Batch Calls

def create_batch_from_jsonl(
    client: OpenAI,
    jsonl_path: Union[str, Path],
    description: str = "Batch task",
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
) -> str:
    """
    Upload the JSONL and create a batch job.
    Returns the batch_id.
    """
    upload = client.files.create(file=open(str(jsonl_path), "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata={"description": description},
    )
    return batch.id


def retrieve_batch(client: OpenAI, batch_id: str):
    """Wrapper to retrieve a batch object."""
    return client.batches.retrieve(batch_id)


def download_batch_output_file(client: OpenAI, file_id: str, save_to: Union[str, Path]) -> Path:
    """
    Given a file_id from a finished batch output, download its content to save_to path.
    """
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)

    content = client.files.content(file_id)
    content_bytes = content.read()
    with save_to.open("wb") as f:
        f.write(content_bytes)
    return save_to


# Output Parsing

def parse_batch_output(jsonl_path: Union[str, Path], task: str) -> Dict[str, Union[str, List[str]]]:
    """
    Parse the batch output JSONL (the one returned by the Batch API).
    Returns a mapping: custom_id -> (string for judgement | list[str] for generation).
    """
    preds: Dict[str, Union[str, List[str]]] = {}

    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # Defensive: skip lines without response or choices (errors, etc.)
            resp = item.get("response", {})
            body = resp.get("body", {})
            choices = body.get("choices", [])

            if not choices:
                continue

            custom_id = item.get("custom_id")
            if not custom_id:
                continue

            if task == TASK_JUDGEMENT:
                preds[custom_id] = choices[0]["message"]["content"]
            else:
                preds[custom_id] = [c["message"]["content"] for c in choices]

    return preds


# Post-Processing

_A_VARIANTS = {
    "A", "A) TRUE", "TRUE", "TRUE (A)", "CORRECT", "YES",
    "A) \\TEXT{TRUE}", "A \\TEXT{) TRUE}", "A \\TEXT{ TRUE}", "A) \\TEXT{TRUE",
    "A)\\ TRUE", "\\TEXT{TRUE", "\\TEXT{A)", "\\TEXT{A", "\\TEXT{A) TRUE", "A). TRUE",
    "A) \\TEXT{TRUE", "A) \\, \\TEXT{TRUE}", "TRUE} \\TEXT{ (A)", "A \\, \\TEXT{TRUE",
}
_B_VARIANTS = {
    "B", "B) FALSE", "FALSE", "FALSE (B)", "INCORRECT", "NO", "B: FALSE",
    "B) \\TEXT{FALSE", "B)\\ FALSE", "B}) \\TEXT{FALSE", "B)} \\TEXT{FALSE", "B)",
    "B \\, \\TEXT{FALSE", "\\TEXT{FALSE", "B \\TEXT{ (FALSE)", "B \\TEXT{ FALSE", "B). FALSE",
    "B \\TEXT{) FALSE", "B) \\, \\TEXT{FALSE", "B} \\TEXT{ (B)", "B} \\TEXT{ FALSE",
}
_C_VARIANTS = {
    "C", "C) UNCERTAIN", "UNCERTAIN", "UNKNOWN", "INDETERMINATE", "CANNOT BE DETERMINED",
    "UNCERTAIN (C)", "C: UNCERTAIN", "C \\TEXT{ (UNCERTAIN", "C) \\ \\TEXT{UNCERTAIN",
    "C)\\, \\TEXT{UNCERTAIN", "C) \\TEXT{UNCERTAIN", "C)\\, UNCERTAIN", "C), UNCERTAIN",
    "C \\TEXT{) UNCERTAIN", "\\TEXT{C", "\\TEXT{C) UNCERTAIN", "\\TEXT{UNCERTAIN", "\\TEXT{uncertain",
    "C)", "C (UNCERTAIN)", "C) \\ UNCERTAIN", "C \\TEXT{ (UNCERTAIN)", "C}) \\TEXT{ UNCERTAIN",
}

def _normalize_answer_token(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).upper()


def correct_options(answer: Union[str, bool]) -> Union[str, bool]:
    """
    Map many textual variants to canonical {A,B,C}.
    Returns input unchanged if no mapping found.
    """
    if isinstance(answer, bool):
        return "A" if answer else "B"

    token = _normalize_answer_token(str(answer))
    if token in _A_VARIANTS:
        return "A"
    if token in _B_VARIANTS:
        return "B"
    if token in _C_VARIANTS:
        return "C"
    return answer


_BOXED_PAT = re.compile(r"\\boxed\{\\text\{(.*?)\}\}|\\boxed\{(.*?)\}")

def extract_boxed_answer(text: str) -> str:
    """
    Extract the content of \boxed{...} (optionally nested \text{...}).
    Then map it to A/B/C with correct_options.
    Returns '' if nothing found.
    """
    if not isinstance(text, str):
        return ""
    m = _BOXED_PAT.search(text)
    if not m:
        return ""
    candidate = m.group(1) or m.group(2) or ""
    mapped = correct_options(candidate)
    return str(mapped).strip()


# Merging & Scoring

def merge_predictions(
    df_train: pd.DataFrame,
    preds: Dict[str, Union[str, List[str]]],
    *,
    request_prefix: str = "request-",
    input_col: str = "question",
    label_col: Optional[str] = "label",
    task: str = TASK_GENERATION,
) -> pd.DataFrame:
    """
    Merge predictions (by custom_id) back into the training DataFrame.
    """
    # Build dataframe of predictions with numeric id extracted from custom_id.
    rows = []
    for cid, val in preds.items():
        try:
            idx = int(str(cid).split(request_prefix, 1)[1])
        except Exception:
            continue
        rows.append({"id": idx, "pred": val})

    df_pred = pd.DataFrame(rows)
    df_train = df_train.copy().reset_index(drop=True)
    df_train["id"] = df_train.index

    df_merged = pd.merge(df_train, df_pred, on="id", how="left").drop(columns=["id"])

    if task == TASK_JUDGEMENT:
        df_merged.rename(columns={"pred": "prediction"}, inplace=True)
    else:
        df_merged.rename(columns={"pred": "reasons"}, inplace=True)

    # Ensure a canonical 'input' column exists for downstream steps
    if input_col in df_merged.columns and "input" not in df_merged.columns:
        df_merged["input"] = df_merged[input_col]

    # If both 'answer' and 'label' exist, keep 'label'; if only 'answer', rename -> 'label'
    if label_col and label_col in df_merged.columns:
        pass
    elif "answer" in df_merged.columns and "label" not in df_merged.columns:
        df_merged.rename(columns={"answer": "label"}, inplace=True)

    return df_merged


def postprocess_generation(df_generation: pd.DataFrame) -> pd.DataFrame:
    """
    For generation task:
      - explode multi-sample reasons
      - extract boxed answers -> predicted_answer in {A,B,C}
      - compute reward if gold label exists
    """
    if "reasons" not in df_generation.columns:
        raise ValueError("Expected 'reasons' column for generation post-processing.")

    # Explode multi-sample outputs into one row per sample
    df_expanded = df_generation.copy()
    df_expanded = df_expanded.explode("reasons").reset_index(drop=True)

    # Extract \boxed{...} -> {A,B,C}
    df_expanded["predicted_answer"] = df_expanded["reasons"].apply(extract_boxed_answer)

    # Keep only A/B/C
    df_expanded = df_expanded[df_expanded["predicted_answer"].isin(["A", "B", "C"])].copy()

    # Reward if label is present
    if "label" in df_expanded.columns:
        df_expanded["reward"] = (df_expanded["label"].astype(str).str.upper().str[0] == df_expanded["predicted_answer"]).astype(int)
    else:
        df_expanded["reward"] = None

    # Add placeholders to match your schema if needed
    if "system" not in df_expanded.columns:
        df_expanded["system"] = ""
    if "reasoning_label" not in df_expanded.columns:
        df_expanded["reasoning_label"] = ""

    return df_expanded


def to_orm(
    df_generation_expanded: pd.DataFrame,
    *,
    input_col: str = "input",
    reasons_col: str = "reasons",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Build an ORM-style dataset from expanded generation rows.
    Keeps rows with non-empty reasons and valid predicted_answer.
    """
    required = [input_col, reasons_col, label_col, "predicted_answer"]
    for col in required:
        if col not in df_generation_expanded.columns:
            raise ValueError(f"Missing required column '{col}' for ORM conversion.")

    df = df_generation_expanded.copy()
    df = df[(df[reasons_col].astype(str).str.strip() != "") & (df["predicted_answer"].isin(["A", "B", "C"]))]

    # Reward is already computed in postprocess_generation (if label present)
    if "reward" not in df.columns:
        df["reward"] = (df[label_col].astype(str).str.upper().str[0] == df["predicted_answer"]).astype(int)

    # Minimal ORM projection (rename to your preferred schema if needed)
    orm = df[[input_col, label_col, reasons_col, "predicted_answer", "reward"]].rename(
        columns={"predicted_answer": "predicted_label"}
    )
    return orm


# I/O Helpers

def save_json(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", indent=4, force_ascii=False)


if __name__ == "__main__":
    DATA_PATH = "/path/to/data.json"                       # input training data
    BATCH_INPUT_JSONL = "/path/to/gpt4o/input.jsonl"       # batch request file
    BATCH_OUTPUT_JSONL = "/path/to/gpt4o/output.jsonl"     # batch result file
    FINAL_OUTPUT_JSON = "/path/to/final/output.json"       # merged or post-processed output
    ORM_OUTPUT_JSON = "/path/to/final/orm_output.json"     # optional ORM export

    #Load data
    df_train = pd.read_json(DATA_PATH)

    # Select task
    TASK = TASK_GENERATION      # or TASK_JUDGEMENT
    cfg = make_task_config(TASK)

    # Build batch input
    create_jsonl_for_chat(
        df_train,
        BATCH_INPUT_JSONL,
        cfg,
        input_col="question",
        label_col="label",
        request_prefix="request-",
    )

    # Submit batch - To be run independently
    # client = OpenAI()
    # batch_id = create_batch_from_jsonl(
    #     client,
    #     BATCH_INPUT_JSONL,
    #     description="Task Name",
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    # )
    # print("Batch ID:", batch_id)
    #
    # # Later, retrieve & download output:
    # batch = retrieve_batch(client, batch_id)
    # # Once finished, find batch.output_file_id and download:
    # output_file_id = batch.output_file_id    # example attribute name
    # download_batch_output_file(client, output_file_id, BATCH_OUTPUT_JSONL)

    # After batch run, where batch output is available
    # Parse batch output
    preds = parse_batch_output(BATCH_OUTPUT_JSONL, TASK)

    # Merge back to train
    df_final = merge_predictions(
        df_train,
        preds,
        request_prefix="request-",
        input_col="question",
        label_col="label",
        task=TASK,
    )

    # Save based on task
    if TASK == TASK_JUDGEMENT:
        # Judgement flow: predictions are "Correct"/"Incorrect"
        save_json(df_final, FINAL_OUTPUT_JSON)

    else:
        # Generation flow: post-process to extract \boxed{...} -> {A,B,C}
        df_expanded = postprocess_generation(df_final)

        # Save the expanded generation results
        save_json(df_expanded, FINAL_OUTPUT_JSON)

        # (Optional) Build and save ORM-style dataset
        orm_df = to_orm(
            df_expanded,
            input_col="input",
            reasons_col="reasons",
            label_col="label",
        )
        save_json(orm_df, ORM_OUTPUT_JSON)
