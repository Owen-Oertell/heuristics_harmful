import argparse
import re
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from accuracy_utils import process_sample, math_verify_check


def compute_score(solution_str, ground_truth):
    retval = 0.
    answer = None
    try:
        # print('raw answer', solution_str)
        answer = re.sub(r"<think>.*?</think>\n?\n?", "", solution_str, flags=re.DOTALL).strip()
        # answer = re.sub(r".*?</think>\n?\n?", "", solution_str, flags=re.DOTALL)
        # print('regexed answer', answer)
        answer = process_sample(answer)
        if math_verify_check(ground_truth, answer):
            retval = 1.
    except Exception as e:
        print(e)

    return answer, retval


# def set_seed(seed=5775709):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--dataset", type=str, default="aime-24")
    parser.add_argument("--maxlen", type=int, default=16384)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--expanded_prompt", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument('--seed', type=int, default=11111)
    parser.add_argument('--use_seed', type=bool, default=False)
    return parser.parse_args()


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


def main():

    # init
    args = parse_arguments()
    print(args)

    # dataset
    dataset = get_dataset(args.dataset)
    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    # add id column
    dataset = dataset.map(
        lambda example, idx: {"message_id": f"id_{idx}"},
        with_indices=True,
    )

    # duplicate dataset
    indices = [i % len(dataset) for i in range(len(dataset) * args.n)]
    dataset = dataset.select(indices)

    if args.expanded_prompt:
        prompts = [f"<｜begin▁of▁sentence｜><｜User｜>{dataset[i]['problem']} Please think step-by-step and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n" for i in tqdm(range(len(dataset)))]
    else:
        prompts = [f"<｜begin▁of▁sentence｜><｜User｜>{dataset[i]['problem']}<｜Assistant｜><think>\n" for i in tqdm(range(len(dataset)))]

    # load model
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.world_size,
    )

    # start generate
    if args.use_seed:
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=-1,
            max_tokens=args.maxlen,
            seed=args.seed,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=-1,
            max_tokens=args.maxlen,
            seed=None,
        )
    response = llm.generate(prompts, sampling_params)
    output = list(map(lambda x: x.outputs[0].text, response))
    dataset = dataset.add_column(f"responses", output)

    # evaluate
    answers, scores = [], []
    for d in tqdm(range(len(dataset))):
        answer, score = compute_score(dataset[d]['responses'], dataset[d]['answer'])
        answers.append(answer)
        scores.append(score)
    scores = np.array(scores)
    dataset = dataset.add_column("processed_answer", answers)
    dataset = dataset.add_column("reward", scores == 1)

    # print(args.model_name)
    # print(args.dataset)
    # print(f"args.expanded_prompt: {args.expanded_prompt}")
    # all_scores = np.array(all_scores)
    # print(f"avg@{args.n}: {all_scores.mean()}")
    # all_scores = np.cumsum(all_scores, axis=0) > 0
    # print(f"pass@{args.n}: {all_scores.mean(axis=1).tolist()}")
    dataset.push_to_hub(f'GitBag/{args.model_name.split("/")[-1]}_{args.dataset}_eval_new_{str(args.n)}')
    print(args.model_name)
    print(args.dataset)
    print(scores.mean())

    examples = []
    message_id_index_d = {}

    for i in range(len(dataset)):
        message_id = dataset[i]['message_id']
        if message_id not in message_id_index_d:
            message_id_index_d[message_id] = len(message_id_index_d)
            examples.append([])
        examples[message_id_index_d[message_id]].append(dataset[i]['reward'])

    print(f"examples: {examples}")
    one_correct = []
    for e in examples:
        if True in e:
            one_correct.append(1)
        else:
            one_correct.append(0)
    print(np.mean(one_correct))


if __name__ == "__main__":
    main()