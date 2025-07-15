import os
import datasets
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':

    # load dataset
    data_source = 'BytedTsinghua-SIA/DAPO-Math-17k'
    train_dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    train_dataset = train_dataset['train']

    # filter out depulicates
    frequency = {}
    index = []
    for i in tqdm(range(len(train_dataset))):
        if train_dataset[i]['prompt'][0]['content'] not in frequency:
            frequency[train_dataset[i]['prompt'][0]['content']] = 1
            index.append(i)
        else:
            frequency[train_dataset[i]['prompt'][0]['content']] += 1

    # select uniques
    train_dataset = train_dataset.select(index)

    # push to hub
    train_dataset.push_to_hub('GitBag/dapo-math-17k')