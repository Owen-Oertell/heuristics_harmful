from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-7B"
PROMPTS = [
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
    "Let $a,$ $b,$ $c,$ $d$ be distinct complex numbers such that $|a| = |b| = |c| = |d| = 1$ and $a + b + c + d = 0.$  Find the maximum value of \[|(a + b)(a + c)(a + d)(b + c)(b + d)(c + d)|.\] Let's think step by step and output the final answer within \\boxed{}.",
]

# load model
llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=1,
    top_p=0.95,
    top_k=-1,
    max_tokens=16384,
    seed=None,
)
response = llm.generate(PROMPTS, sampling_params)
output = list(map(lambda x: x.outputs[0].text, response))

for i in output:
    print('=' * 100)
    print(i)