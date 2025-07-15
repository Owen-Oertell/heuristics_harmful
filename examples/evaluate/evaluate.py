from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from examples.bon_model_test.reward_score import math, math_dapo

import argparse
import os
import torch
import random
import numpy as np
import grader

def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GitBag/a_star_final_a_star_math_7_actor")
    parser.add_argument("--dataset", type=str, default="minervamath")
    parser.add_argument("--type", type=str, default="exp")
    parser.add_argument("--maxlen", type=int, default=16384)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    return parser.parse_args()


def format_answer(example):
    new_answer = example["answer"][0].strip("$")
    return {"answer": new_answer}


def get_dataset(name):
    # return a dataset with columns "problem" and "answer"
    if name == "aime-24":
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        dataset = dataset.rename_column("Problem", "problem").rename_column("Answer", "answer")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "aime-25":
        dataset = load_dataset("yentinglin/aime_2025", split="train")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "hmmt-feb-24":
        dataset = load_dataset("MathArena/hmmt_feb_2024", split="train")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "hmmt-feb-25":
        dataset = load_dataset("MathArena/hmmt_feb_2025", split="train")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "amc-23":
        dataset = load_dataset("math-ai/amc23", split="test")
        dataset = dataset.rename_column("question", "problem")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "minervamath":
        dataset = load_dataset("math-ai/minervamath", split="test")
        dataset = dataset.rename_column("question", "problem")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        return dataset
    elif name == "olympiadbench":
        dataset = load_dataset("math-ai/olympiadbench", split="test")
        dataset = dataset.rename_column("question", "problem").rename_column("final_answer", "answer")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["problem", "answer"]])
        dataset = dataset.map(format_answer)
        return dataset

    raise NotImplementedError


def apply_math_template(example):
    # apply your function to the column "col_name"
    example["problem"] = example["problem"] + " Let's think step by step and output the final answer within \\boxed{}."
    return example


def apply_dapo_template(example):
    # apply your function to the column "col_name"
    example["problem"] = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n" + example["problem"] + "\n\nRemember to put your answer on its own line after \"Answer:\""
    return example


def main():

    # init
    args = parse_arguments()

    # dataset
    dataset = get_dataset(args.dataset)
    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    # modify prompt based on type
    if args.type == 'math':
        dataset = dataset.map(apply_math_template)
        compute_score = math.compute_score
    elif args.type == 'exp':
        dataset = dataset.map(apply_math_template)
        compute_score = grader.compute_score
    elif args.type == 'dapo':
        dataset = dataset.map(apply_dapo_template)
        compute_score = math_dapo.compute_score
    else:
        raise NotImplementedError

    # load model
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.world_size,
    )

    # start generate
    for p in range(args.n):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            top_k=-1,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate([dataset[i]['problem'] for i in range(len(dataset))], sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        dataset = dataset.add_column(f"response_{p}", output)

    # evalaute
    all_scores = []

    for i in tqdm(range(args.n)):
        scores = []
        for d in range(len(dataset)):
            if args.type == 'math':
                scores.append(compute_score(dataset[d][f'response_{i}'], str(dataset[d]['answer']))[1])
            elif args.type == 'dapo':
                scores.append(compute_score(dataset[d][f'response_{i}'], str(dataset[d]['answer']))['acc'])
            elif args.type == 'exp':
                scores.append(compute_score(dataset[d][f'response_{i}'], str(dataset[d]['answer'])))
            else:
                raise NotImplementedError
        scores = np.array(scores)
        all_scores.append(scores)
        dataset = dataset.add_column(f"eval_{i}", scores)

    print(args.model_name)
    print(args.dataset)
    all_scores = np.array(all_scores)
    print(f"avg@{args.n}: {all_scores.mean()}")
    all_scores = np.cumsum(all_scores, axis=0) > 0
    print(f"pass@{args.n}: {all_scores.mean(axis=1).tolist()}")
    # dataset.push_to_hub(f'GitBag/{args.model_name.split("/")[-1]}_{args.dataset}_eval')


if __name__ == "__main__":
    main()