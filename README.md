# Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation

This repository contains the code for the paper "Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation". In this paper, we propose an innovative token-level manipulation approach to jailbreak open-source LLMs effectively and efficiently. Our paper is released on [Arxiv](https://arxiv.org/abs/2405.13068).


## Introduction

Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JAILMINE, an innovative token-level manipulation approach that addresses these limitations effectively.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/LLM-Integrity-Guard/JailMine.git
cd JailMine
pip install -r requirements.txt
```

## Dataset and Models

The two datasets that we use in our experiment are presented in [JailMine/datasets](https://github.com/LLM-Integrity-Guard/JailMine/tree/main/datasets). Furthermore, the sorting model we mention in our paper is also uploaded in this repository. 

## Usage

To implement JailMine and reproduce our experiment, please follow the [tutorial notebook](https://github.com/LLM-Integrity-Guard/JailMine/blob/main/Tutorial.ipynb). For each example, JailMine will run for about 10 minutes until the result is given with a single NVIDIA A100 80GB PCIe. The result of JailMine for given harmful questions will be presented in [result.csv](https://github.com/LLM-Integrity-Guard/JailMine/blob/main/result.csv).

## Reproducibility

To reproduce the result in our paper, please run the following script:

```bash
HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> python reproduce.py
```
Replace `<YOUR_HUGGINGFACE_TOKEN>` with your actual Huggingface token.

The explaination of each parameter is described as follows:

```
--MODEL_NAME     # The name of the target model
--MODEL_PATH     # The local path of the target model
--REPHRASE_PATH  # The local path of the rephrase model. We recommand Llama-2-7b-chat as the rephrase model.
--SORTING_PATH   # The local path of the sorting model.
--EMBED_PATH     # The local path of the embed model. We recommand gtr-t5-xl as the embed model.
--JUDGE_PATH     # The local path of the judge model. We recommand Llama-Guard-2-8b as the judge model.
# Please modify these paths to our own paths
--device         # The device you want to use when reproducing.
--n_devices      # The number of devices you want to use. If n_devices > 1, "device" will be abandoned.
--question_path  # The local path of the question set.
--N              # The number of manipulation you want to make. Default is 2000.
--m              # The length of manipulation after the first k manipulation. Default is 5.
--n              # The number of responses to each harmful question. Default is 1.
```

