
# WebNLG-Interno

This repo contains data and code to reproduce our experiments for [WebNLG-2023 Challenge](https://synalp.gitlabpages.inria.fr/webnlg-challenge/challenge_2023/).
The code is based on Hugging Face [🤗Transformers](https://huggingface.co/docs/transformers/index) and [PEFT](https://huggingface.co/docs/peft/index).

# Setup

## Pre-requirements

The code was tested under `python3.9 & CUDA 11.7`

## Installation

1. Clone repository

   ```shell
   git clone webnlg_interno.git
   cd webnlg_interno
   ```

2. Create virtual environment

   ```shell
   python3 -m venv ./interno_env
   source ./interno_env/bin/activate
   pip install --upgrade pip
   ```

3. Within virtual environment run:

   ```shell
   pip install -r requirements.txt
   ```

4. Install METEOR

   ```shell
   cd metrics/webnlg_2023/evaluation/automatic/scripts
   bash install_dependencies.sh
   ```

# Data

All files related to data processing are located in the `data` folder:

* `original datasets` contains original XML files and corresponding JSON files for convenience. Script  `xml_to_json.py` is used for conversion.
* `refs` contains reference files for _dev_ and _test_ splits ready to use for [automatic evaluation](https://github.com/WebNLG/2023-Challenge/tree/main/evaluation/automatic/scripts). Multiple references are generated by [these guidelines](https://github.com/WebNLG/GenerationEval#multiple-references) (but just omitted "a-" like prefix). Script `generate_refs.py` is used for generation.
* `created_datasets` contains datasets for training, validation and testing for different prompt construction strategies (simple, with_links and full).  
   - To create a __simple__ dataset run:
      ```shell
      python nlg_data.py --file <path_to_the_json_file>
      ```

   - To create a dataset __with_links__ run:
      ```shell
      python nlg_data.py --file <path_to_the_json_file> --add_links
      ```

   - To create a __full__ dataset with links and metadata run:
      ```shell
      python nlg_data.py --file <path_to_the_json_file> --add_links --add_metadata
      ```
   - To use predicates and categories translated to Russian add flag `--translate`.  
      The generated results slightly differ from the provided (which has been used for experiments) since by the time of challenge we used a web version of translation engine for several examples. Nevertheless, these differences are minor and we do not expect it to significantly affect reproducibility.
   - By default data is saved to `created_datasets`, it can be changed by `--target_dir` argument.
   - __Multi-reference:__ Since entries in the dataset may have more than 1 completion, every completion is treated as a unique sample during training. During validation multi-reference evaluation is performed. To avoid duplicates on inference, we generate additional `*_inference.jsonl` files with a unique occurrence of each entry.

# Pretrained models

Download following pretrained models and locate it in `webnlg_interno/models` folder.

## FRED-T5

Pretrained FRED_T5_1.7B model from hugging face: https://huggingface.co/ai-forever/FRED-T5-1.7B

## mT5 models

Pretrained mT5 models from hugging face:

* mT5-Large (1.2B): https://huggingface.co/google/mt5-large
* mT5-XL (3.7B): https://huggingface.co/google/mt5-xl

# Submission
You can find the checkpoint used for our submission in the `submission` folder.  
To reproduce submitted results run:
```shell
cd webnlg_interno

export CUDA_VISIBLE_DEVICES=0,1

# run inference on test split
bash predict.sh
```

# Training

To reproduce the results submitted to WebNLG-2023:

```shell
cd webnlg_interno

export CUDA_VISIBLE_DEVICES=0,1,2,3

# run training
bash run_summarization.sh
```

To launch your own training or reproduce experiments from the paper, you need to change following arguments in `run_summarization.sh`:

* OUTPUT_DIR - experiment directory
* MODEL_PATH - path to pretrained model (like FRED or mT5)
* TRAIN_FILE, VAL_FILE, TEST_FILE - datasets  

We have run our experiments on `4xV100` GPUs with `total_batch_size = #GPUs * per_device_batch_size * accumulation_steps = 16`.  
For `mT5-XL` experiments we set `--per_device_train_batch_size=1` and `--gradient_accumulation_steps=4` to be able to fit into V100 and keep `total_batch_size = 16`.

__NB:__ During training only the best checkpoint (by METEOR value) and last checkpoint are saved.
## Warning

Temporary files are created for metrics evaluation during validation steps. It is not recommended to perform more than 1 experiments from the same repository folder simultaneously. This may affect correctness of validation.
