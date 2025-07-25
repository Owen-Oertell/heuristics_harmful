# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import numpy as np

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string_preprocessing


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string_preprocessing(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/n/holylabs/LABS/kdbrantley_lab/Lab/zhaolin/exploration/data/math500')
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'xDAN2099/lighteval-MATH'
    train_dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    train_dataset = train_dataset['train']

    data_source = 'HuggingFaceH4/MATH-500'
    test_dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    test_dataset = test_dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('problem')

            question = question_raw + ' ' + instruction_following

            answer = example.pop('solution')
            solution = extract_solution(answer)
            data = {
                "data_source": 'lighteval/MATH',
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # test_dataset = test_dataset.select(np.random.choice(len(test_dataset), args.test_size, replace=False))

    print(f"train_dataset: {train_dataset}")
    print(f"train_dataset[0]: {train_dataset[0]}")
    print(f"test_dataset: {test_dataset}")
    print(f"test_dataset[0]: {test_dataset[0]}")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
