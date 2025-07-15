import re
from accuracy_utils import process_sample
from inference_eval import compute_correctness

def compute_score(solution_str, ground_truth, rm_normalize) -> float:

    if rm_normalize:
        retval = -0.5
    else:
        retval = 0.
    try:
        answer = re.sub(r"<think>.*?</think>\n?\n?", "", solution_str, flags=re.DOTALL).strip()
        answer = process_sample(answer)
        if compute_correctness(ground_truth, answer):
            if rm_normalize:
                retval = 0.5
            else:
                retval = 1.
    except Exception as e:
        print(e)

    return 0, retval