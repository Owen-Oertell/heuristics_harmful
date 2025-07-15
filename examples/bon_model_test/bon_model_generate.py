from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse
import os
import torch
import random
import numpy as np


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="3")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--output_dir", type=str, default="./examples/bon_model_test/")
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--world_size", type=int, default=1)
    return parser.parse_args()


def main():

    # init
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/Qwen2.5-{args.model_size}B")
    llm = LLM(
        model=f"Qwen/Qwen2.5-{args.model_size}B",
        tensor_parallel_size=args.world_size,
    )

    # dataset
    if args.dataset == 'math':
        dataset = load_dataset("parquet", data_files={'test': f"/n/holylabs/LABS/kdbrantley_lab/Lab/zhaolin/exploration/data/math/test.parquet"}, split='test')
    else:
        dataset = load_dataset("parquet", data_files={'test': f'/n/holylabs/LABS/kdbrantley_lab/Lab/zhaolin/reasoning/data/{args.dataset}/test.parquet'}, split='test')
    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    # start generate
    for p in range(args.n):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate([dataset[i]['prompt'][0]['content'] for i in range(len(dataset))], sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        dataset = dataset.add_column(f"response_{p}", output)

    # save
    dataset.save_to_disk(os.path.join(args.output_dir, f'{args.dataset}_size_{args.model_size}'))


if __name__ == "__main__":
    main()