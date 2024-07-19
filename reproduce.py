import JailMine
import torch
import torch.nn as nn
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--MODEL_NAME", type=str, default='mistral-7b-instruct', choices=["Llama-2-7b-chat", "mistral-7b-instruct", "meta-llama/Meta-Llama-3-8B", "gemma-7b-it"])
    parser.add_argument("--MODEL_PATH", type=str, default=f'/data/yichuan/cyc/llm_model/hub/mistral-7b/snapshots/_')
    parser.add_argument("--REPHRASE_PATH", type=str, default='/data/yichuan/cyc/llm_model/hub/Llama-2-7b-chat-hf/snapshots/_' )
    parser.add_argument("--SORTING_PATH", type=str, default='sorting.pth')
    parser.add_argument("--EMBED_PATH", type=str, default='/data/yichuan/cyc/llm_model/hub/gtr-t5-xl/snapshots/_')
    parser.add_argument("--JUDGE_PATH", type=str, default='/data/yichuan/cyc/llm_model/hub/Llama-Guard-2-8B/snapshots/_')
    # Please modify these paths to our own paths
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--question_path", type=str, default="datasets/jailbreakBench.csv") # choose to evaluate advBench or jailbreakBench
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n", type=int, default=1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    MODEL_NAME = args.MODEL_NAME
    MODEL_PATH = args.MODEL_PATH
    REPHRASE_PATH = args.REPHRASE_PATH
    SORTING_PATH = args.SORTING_PATH
    EMBED_PATH = args.EMBED_PATH
    JUDGE_PATH = args.JUDGE_PATH

    

    question_path = args.question_path
    questions_set = pd.read_csv(question_path)['goal'].tolist()

    miner = JailMine.JailMine(model_name=MODEL_NAME,
                              target_model_path=MODEL_PATH,
                              rephrase_model_path=REPHRASE_PATH,
                              sorting_model_path=SORTING_PATH,
                              embedding_model_path=EMBED_PATH,
                              judge_model_path=JUDGE_PATH,
                              device=args.device, # if n_devices > 1, ignore this parameter
                              n_devices=args.n_devices, # number of devices
                             )
    miner.run(questions = questions_set[0:1],
           m=args.m,
           N=args.N,
           n=args.n)

    miner.evaluate_ASR()