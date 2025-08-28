# Logical Reasoning with Outcome Reward Models for Test-Time Scaling
We present a set of Outcome Reward Models (ORMs) for deductive logical reasoning. We propose a novel tactic to expand the type of errors covered in the training dataset of the ORM. In particular, we propose an echo generation technique that leverages LLMs' tendency to reflect incorrect assumptions made in prompts to extract additional training data, covering previously unexplored error types. While a standard CoT chain may contain errors likely to be made by the reasoner, the echo strategy deliberately steers the model toward incorrect reasoning.

![LogicORM](plots/orm-method.pdf)

## Resources

👉 **[Hugging Face Collection](https://huggingface.co/collections/ramyakeerthyt/qwen25-logic-orm-685cd7da3509631fc93235de)**  
👉 **[ArXiv Preprint](http://arxiv.org/abs/2508.19903)**


## Installation

1. **Clone this repository**
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/RamyaKeerthy/LogicORM

2. **Set Up the Environment**
  Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Data Generation
CoT and Echo data are generated using both GPT-4o and Qwen models.
Examples of generated FOLIO data are provided in the data/ directory.
1. **Using GPT4o**
  ```
  src/data_generation_gpt.py
  ```
2. **Using Qwen**
  ```
  src/data_generation.py
  ```

## Finetuning ORM
ORMs finetuned on generated data (both Echo and CoT) are available on [Hugging Face Collection](https://huggingface.co/collections/ramyakeerthyt/qwen25-logic-orm-685cd7da3509631fc93235de)
```
scripts/finetune_orm.sh
```

## Inference ORM
During inference, we use:
- good_token = '+'
- bad_token = '-'
- step_tag = '<extra_0>'
to calculate the likelihood of a reasoning chain.
  ```
  scripts/inference_orm.sh
  ```
## Visualization
Plots and evaluation figures can be generated with:
  ```
  plots/orm_plots.py
  ```

## Citation
If you use this repository, please cite:

Thatikonda, R. K., Buntine, W., & Shareghi, E. (2025). *Logical reasoning with outcome reward models for test-time scaling*. arXiv. https://arxiv.org/abs/2508.19903
