import json
import argparse
import os
import pandas as pd
from collections import Counter

from generate_acts import load_model


def sample_data(model_name, dataset, device):
    # prepare data and prompt
    data_directory = os.path.join('datasets', f"{dataset}.csv")
    df = pd.read_csv(data_directory)
    model = load_model(model_name, device=device)
    length_list = []
    if model_name in ['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft']:
        model_family = 'llama3'
    elif model_name in ['mistral-7B', 'mistral-7B-instruct', 'mistral-7B-SFT']:
        model_family = 'mistral'
    elif model_name in ['deepseek-V2-lite', 'deepseek-V2-lite-chat']:
        model_family = 'deepseek-V2'
    else:
        raise ValueError(f"Model name {model_name} not recognized.")
    
    if args.dataset == 'tulu_extracted':  # This dataset is special. Different pairs have different lengths.
        true_prompts, false_prompts = [], []
        for i in range(0, len(df), 2):
            id1, statement1, label1 = df.iloc[i]
            id2, statement2, label2 = df.iloc[i + 1]
            assert label1 == 1 and label2 == 0
            if len(model.tokenizer(statement1).input_ids) != len(model.tokenizer(statement2).input_ids):
                continue  # Skip pairs with different lengths
            true_prompts.append(statement1)
            false_prompts.append(statement2)

    else:
        true_statements, false_statements = [], []
        for idx, row in df.iterrows():  # We use the most common length and pattern to filter the statements
            tokens = model.tokenizer(row['statement']).input_ids  # tokenize the statement
            # print(tokens)
            length_list.append(len(tokens))
            
            if model_family == 'llama3':
                if args.dataset == 'cities':
                    # Pattern: [Beginning token] / the / city / of/ [city name of 3 tokens] / is / in / [country name of 1 token] / .
                    if not (len(tokens) == 11 and tokens[0:4] == [128000, 791, 3363, 315] and \
                            tokens[7:9] == [374, 304] and tokens[10] == 13):
                        continue

                if args.dataset == 'neg_cities':
                    # Pattern: [Beginning token] / the / city / of/ [city name of 3 tokens] / is / not / in / [country name of 1 token] / .
                    if not (len(tokens) == 12 and tokens[0:4] == [128000, 791, 3363, 315] and \
                            tokens[7:10] == [374, 539, 304] and tokens[11] == 13):
                        continue
                
                if args.dataset == 'larger_than':
                    # Pattern: [Beginning token] / [number of 3 tokens] / is / larger / than / [number of 2 tokens] / .
                    if not(len(tokens) == 10 and tokens[4:7] == [374, 8294, 1109] and tokens[9] == 13):
                        continue
                
                if args.dataset == 'smaller_than':
                    # Pattern: [Beginning token] / [number of 3 tokens] / is / smaller / than / [number of 2 tokens] / .
                    if not(len(tokens) == 10 and tokens[4:7] == [374, 9333, 1109] and tokens[9] == 13):
                        continue
                
                if args.dataset == 'sp_en_trans':
                    # Pattern: [Beginning token] / The / Spanish / word / ' / [Spanish word of 2 tokens] / ' / means / ' / [English word of 1 token] / '.
                    if not(len(tokens) == 12 and tokens[0:5] == [128000, 791, 15506, 3492, 364] and \
                            tokens[7:10] == [6, 3445, 364] and tokens[11] == 4527):
                        continue
                
                if args.dataset == 'neg_sp_en_trans':
                    # Pattern: [Beginning token] / The / Spanish / word / ' / [Spanish word of 2 tokens] / ' / does / not / mean / ' / [English word of 1 token] / '.
                    if not(len(tokens) == 14 and tokens[0:5] == [128000, 791, 15506, 3492, 364] and \
                            tokens[7:12] == [6, 1587, 539, 3152, 364] and tokens[13] == 4527):
                        continue
            
            elif model_family == 'mistral':  # For mistral-7B and mistral-7B-instruct, the beginning token is 1
                # For mistral-7B-SFT, the beginning token is 32768
                if args.dataset == 'cities':
                    # Pattern: [Beginning token] / the / city / of/ [city name of 3 tokens] / is / in / [country name of 1 token] / .
                    if not (len(tokens) == 11 and tokens[1:4] == [1183, 3758, 1070] and \
                            tokens[7:9] == [1117, 1065] and tokens[10] == 29491):
                        continue
                
                if args.dataset == 'neg_cities':
                    # Pattern: [Beginning token] / the / city / of/ [city name of 3 tokens] / is / not / in / [country name of 1 token] / .
                    if not (len(tokens) == 12 and tokens[1:4] == [1183, 3758, 1070] and \
                            tokens[7:10] == [1117, 1227, 1065] and tokens[11] == 29491):
                        continue
                
                if args.dataset == 'larger_than':
                    # Pattern: [Beginning token] / [number of 4 tokens] / is / larger / than / [number of 3 tokens] / .
                    if not(len(tokens) == 12 and tokens[5:8] == [1117, 6852, 1589] and tokens[11] == 29491):
                        continue
                
                if args.dataset == 'smaller_than':
                    # Pattern: [Beginning token] / [number of 4 tokens] / is / smaller / than / [number of 4 tokens] / .
                    if not(len(tokens) == 13 and tokens[5:8] == [1117, 7768, 1589] and tokens[12] == 29491):
                        continue
                
                if args.dataset == 'sp_en_trans':
                    # Pattern: [Beginning token] / The / Spanish / word / ' / [Spanish word of 2 tokens] / ' / means / ' / [English word of 1 token] / '.
                    if not(len(tokens) == 12 and tokens[1:5] == [1183, 10945, 2475, 1232] and \
                            tokens[7:10] == [29510, 3593, 1232] and tokens[11] == 4903):
                        continue
                
                if args.dataset == 'neg_sp_en_trans':
                    # Pattern: [Beginning token] / The / Spanish / word / ' / [Spanish word of 2 tokens] / ' / does / not / mean / ' / [English word of 1 token] / '.
                    if not(len(tokens) == 14 and tokens[1:5] == [1183, 10945, 2475, 1232] and \
                            tokens[7:12] == [29510, 2003, 1227, 2840, 1232] and tokens[13] == 4903):
                        continue
            
            elif model_family == 'deepseek-V2':
                if args.dataset == 'cities':
                    # Pattern: [Beginning token] / the / city / of/ [city name of 3 tokens] / is / in / [country name of 1 token] / .
                    if not (len(tokens) == 11 and tokens[0:4] == [100000, 549, 3787, 280] and \
                            tokens[7:9] == [317, 279] and tokens[10] == 13):
                        continue
                
                if args.dataset == 'neg_cities':
                    # Pattern: [Beginning token] / the / city / of/ [city name of 3 tokens] / is / not / in / [country name of 1 token] / .
                    if not (len(tokens) == 12 and tokens[0:4] == [100000, 549, 3787, 280] and \
                            tokens[7:10] == [317, 441, 279] and tokens[11] == 13):
                        continue
                
                if args.dataset == 'larger_than':
                    # Pattern: [Beginning token] / [number of 4 tokens] / is / larger / than / [number of 3 tokens] / .
                    if not(len(tokens) == 12 and tokens[0] == 100000 and tokens[5:8] == [317, 5579, 853] and tokens[11] == 13):
                        continue
                
                if args.dataset == 'smaller_than':
                    # Pattern: [Beginning token] / [number of 4 tokens] / is / smaller / than / [number of 3 tokens] / .
                    if not(len(tokens) == 12 and tokens[0] == 100000 and tokens[5:8] == [317, 6611, 853] and tokens[11] == 13):
                        continue
                
                if args.dataset == 'sp_en_trans':
                    # Pattern: [Beginning token] / The / Spanish / word / ' / [Spanish word of 2 tokens] / ' / means / ' / [English word of 1 token] / '.
                    if not(len(tokens) == 12 and tokens[0:5] == [100000, 549, 12299, 1734, 655] and \
                            tokens[7:10] == [6, 2456, 655] and tokens[11] == 6767):
                        continue
                
                if args.dataset == 'neg_sp_en_trans':
                    # Pattern: [Beginning token] / The / Spanish / word / ' / [Spanish word of 2 tokens] / ' / does / not / mean / ' / [English word of 1 token] / '.
                    if not(len(tokens) == 14 and tokens[0:5] == [100000, 549, 12299, 1734, 655] and \
                            tokens[7:12] == [6, 1217, 441, 2059, 655] and tokens[13] == 6767):
                        continue
            
            if row["label"] == 1:
                true_statements.append(row["statement"])
            else:
                false_statements.append(row["statement"])
        
        # print(Counter(length_list))
        # exit()
        # print(f'After filtering token positions, we have {len(true_statements)} true statements and {len(false_statements)} false statements.')
        # exit()

        false_by_last_word = {}
        for statement in false_statements:  # match each true statement with a false statement that has the same last word
            last_word = statement.strip().split()[-1]
            false_by_last_word.setdefault(last_word, []).append(statement)
        
        false_prompts, true_prompts = [], []
        for true_statement in true_statements:
            last_word = true_statement.strip().split()[-1]
            # Check if there's any false statement with the same last word
            if last_word in false_by_last_word:
                if len(false_by_last_word[last_word]) > 1:
                    false_statement = false_by_last_word[last_word].pop(-1)  # remove it from the pool
                else:
                    false_statement = false_by_last_word[last_word][0]
                false_prompts.append(false_statement)
                true_prompts.append(true_statement)
            else:
                if args.dataset == 'larger_than':
                    assert last_word == 'fifty-one.'
                elif args.dataset == 'smaller_than':
                    assert last_word == 'eighty-nine.' or last_word[:7] == 'ninety-'

    print(f"After matching false with true statements, we have {len(false_prompts)} false prompts and {len(true_prompts)} true prompts.")    
    # Save them in a json file without influencing the original dataset
    if os.path.exists(f'datasets/{dataset}_paired.json'):
        with open(f'datasets/{dataset}_paired.json', 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[f'{model_family}_false_prompts'] = false_prompts
    data[f'{model_family}_true_prompts'] = true_prompts
    with open(f'datasets/{dataset}_paired.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistral-7B', choices=[
        'llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft',
        'mistral-7B', 'mistral-7B-instruct', 'mistral-7B-SFT',
        'deepseek-V2-lite', 'deepseek-V2-lite-chat'
    ])
    parser.add_argument('--dataset', type=str, default='tulu_extracted')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    print(args)
    sample_data(args.model, args.dataset, args.device)
