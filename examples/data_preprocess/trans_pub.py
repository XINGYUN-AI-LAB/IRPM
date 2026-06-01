
import argparse
import os
import sys
import json

import datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/verl_steer3filter_prompt4_x2_score')
    parser.add_argument('--data_source', default='public')
    parser.add_argument('--ability', default='public')
    parser.add_argument('dataset', nargs='?', default='./data/steer3filter_prompt4_x2_score')  # 输入文件夹
    
    args = parser.parse_args()
    dataset = datasets.load_from_disk(args.dataset, keep_in_memory=True)
    # dataset = datasets.load_from_disk(args.dataset)
    
    print(dataset)
    print(dataset['train'][0])
    print(dataset['train'][1])
    print(json.dumps(dataset['train'][10], indent=1, separators=(':', ','), ensure_ascii=False))
    print(json.dumps(dataset['train'][100], indent=1, separators=(':', ','), ensure_ascii=False))


    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    data_source = args.data_source
    print(data_source)
    ability = args.ability

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop('problem')
            prompt = problem
            answer = example.pop('solution')
            # if len(problem) > 3500:
            #     print(f"Too Long {len(problem)}, <<{problem}>>")
            #     return None


            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)
    print(train_dataset[0])
    output_dir = args.output_dir
    train_dataset.to_parquet(os.path.join(output_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(output_dir, 'test.parquet'))

if __name__ == '__main__':
    main()