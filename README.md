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
HF_TOKEN = <YOUR_HUGGINGFACE_TOKEN> python reproduce.py
```
Replace `<YOUR_HUGGINGFACE_TOKEN>` with your actual Huggingface token.

