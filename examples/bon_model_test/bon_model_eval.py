from datasets import load_from_disk
from tqdm import tqdm
from reward_score import gsm8k, math, multiply, countdown
import argparse
import numpy as np


def _select_rm_score_fn(reward_function):
    if reward_function == 'gsm8k':
        return gsm8k.compute_score
    elif reward_function == 'math':
        return math.compute_score
    elif "multiply" in reward_function or "arithmetic" in reward_function:
        return multiply.compute_score
    elif "countdown" in reward_function:
        return countdown.compute_score
    else:
        raise NotImplementedError


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./examples/bon_model_test/math_size_1.5")
    parser.add_argument("--reward_function", type=str, default="math")
    parser.add_argument("--n", type=int, default=32)
    return parser.parse_args()


def evaluate(dataset, reward_name, reward_function, n):

    all_scores, all_fs_scores = [], []
    for d in range(len(dataset)):
        scores, fs_scores = [], []
        for i in range(n):
            if reward_name == 'gsm8k':
                fs = reward_function(dataset[d][f'response_{i}'], dataset[d]['reward_model']['ground_truth'], method='strict')[-1]
                cs = reward_function(dataset[d][f'response_{i}'], dataset[d]['reward_model']['ground_truth'], method='flexible')[-1]
            else:
                fs, cs = reward_function(dataset[d][f'response_{i}'], dataset[d]['reward_model']['ground_truth'])
            fs_scores.append(fs)
            scores.append(cs)
        all_scores.append(scores)
        all_fs_scores.append(fs_scores)
    all_scores = np.array(all_scores)
    all_scores = np.cumsum(all_scores, axis=1) > 0
    print(all_scores.mean(axis=0))
    all_fs_scores = np.array(all_fs_scores)
    all_fs_scores = np.cumsum(all_fs_scores, axis=1) > 0
    print(all_fs_scores.mean(axis=0))


def main():

    # init
    args = parse_arguments()

    # dataset
    dataset = load_from_disk(args.dataset)

    # reward function
    reward_function = _select_rm_score_fn(args.reward_function)

    # evaluate
    evaluate(dataset, args.reward_function, reward_function, args.n)

if __name__ == "__main__":
    main()