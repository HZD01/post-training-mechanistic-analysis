import json
import argparse
from tqdm import tqdm
import time
import numpy as np
import random
import re
import torch
import os
import difflib

from generate_acts import load_model


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For complete reproducibility, especially with convolutional operations, add:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(model_name, dataset, classes, need_question):
    if model_name in ['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft']:
        model_family = 'llama3'
    elif model_name in ['llama-2-13b', 'llama-2-13b-instruct']:
        model_family = 'llama2'
    elif model_name in ['mistral-7B', 'mistral-7B-instruct', 'mistral-7B-SFT']:
        model_family = 'mistral'
    elif model_name in ['deepseek-V2-lite', 'deepseek-V2-lite-chat']:
        model_family = 'deepseek-V2'
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    with open(f"datasets/{dataset}_paired.json", 'r') as f:
        data = json.load(f)
    true_prompts = data[f'{model_family}_true_prompts']
    false_prompts = data[f'{model_family}_false_prompts']

    set_random_seed(1)
    print(f'Loaded {len(true_prompts)} pairs of true/false prompts')

    if dataset == 'cities':
        example_true_list = ["The city of Dar es Salaam is in Tanzania.", "The city of Kozhikode is in India."]
        example_false_list = ["The city of Dar es Salaam is in Italy.", "The city of Kozhikode is in the United States."]
    elif dataset == 'neg_cities':
        example_false_list = ["The city of Dar es Salaam is not in Tanzania.", "The city of Kozhikode is not in India."]
        example_true_list = ["The city of Dar es Salaam is not in Italy.", "The city of Kozhikode is not in the United States."]
    elif dataset == 'larger_than':
        example_false_list = ["Fifty-eight is larger than ninety-six.", "Seventy-nine is larger than ninety-seven."]
        example_true_list = ["Seventy-eight is larger than seventy-three.", "Ninety-six is larger than sixty-six."]
    elif dataset == 'smaller_than':
        example_true_list = ["Fifty-eight is smaller than ninety-six.", "Seventy-nine is smaller than ninety-seven."]
        example_false_list = ["Seventy-eight is smaller than seventy-three.", "Ninety-six is smaller than sixty-six."]
    elif dataset == 'sp_en_trans':
        example_true_list = ["The Spanish word 'bosque' means 'forest'.", "The Spanish word 'piel' means 'skin'."]
        example_false_list = ["The Spanish word 'gobernar' means 'to eat'.", "The Spanish word 'edad' means 'clock'."]
    elif dataset == 'neg_sp_en_trans':
        example_false_list = ["The Spanish word 'bosque' does not mean 'forest'.", "The Spanish word 'piel' does not mean 'skin'."]
        example_true_list = ["The Spanish word 'gobernar' does not mean 'to eat'.", "The Spanish word 'edad' does not mean 'clock'."]
    elif dataset == 'tulu_extracted':
        example_true_list = ["The Eiffel Tower is located in Paris.", "'The Great Gatsby' was written by F. Scott Fitzgerald."]
        example_false_list = ["The largest moon of Saturn is Earth.", "Albert Einstein developed the theory of evolution."]
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized")

    if need_question:   # Add "This statement is:" at the end of the prompt
        if classes in ['true_false', 'true_false_noise']:
            example_list = [f"{example} This statement is: TRUE\n" for example in example_true_list]
            example_list.extend([f"{example} This statement is: FALSE\n" for example in example_false_list])
        else:
            raise ValueError(f"Class {classes} not recognized")        
    else:
        example_list = [f"{example}\n" for example in example_true_list]

    final_true_prompts = []
    final_false_prompts = []
    for i in range(len(true_prompts)):
        random.shuffle(example_list)  # randomly perturb the order of the examples
        if need_question:
            final_true_prompts.append(''.join(example_list) + true_prompts[i] + " This statement is:")
            final_false_prompts.append(''.join(example_list) + false_prompts[i] + " This statement is:")
        else:
            splitted_true_prompt = true_prompts[i].split()
            label = splitted_true_prompt[-1][:-1]   # remove the last character (the period)
            actual_true_prompt = ' '.join(splitted_true_prompt[:-1])
            splitted_false_prompt = false_prompts[i].split()
            assert label == splitted_false_prompt[-1][:-1]
            actual_false_prompt = ' '.join(splitted_false_prompt[:-1])
            if dataset == 'sp_en_trans':
                actual_true_prompt = actual_true_prompt + " '"
                actual_false_prompt = actual_false_prompt + " '"
                label = label[1:-1]  # remove the single quotes
            final_true_prompts.append([''.join(example_list) + actual_true_prompt, label])
            final_false_prompts.append([''.join(example_list) + actual_false_prompt, label])
    true_prompts = final_true_prompts
    false_prompts = final_false_prompts
    
    if model_name == 'mistral-7B-SFT':   # Fix its tokenizer's bug
        if need_question:
            true_prompts = ['<s> ' + prompt for prompt in true_prompts]
            false_prompts = ['<s> ' + prompt for prompt in false_prompts]
        else:
            true_prompts = [['<s> ' + prompt[0], prompt[1]] for prompt in true_prompts]
            false_prompts = [['<s> ' + prompt[0], prompt[1]] for prompt in false_prompts]
    
    return false_prompts, true_prompts


# Calculate the model’s average likelihood of the two classes.
def check_prob_one_model(model_name, dataset, device, classes, need_question):
    assert len(model_name) == 1
    false_prompts, true_prompts = load_data(model_name[0], dataset, classes, need_question)
    model = load_model(model_name[0], device=device)
    f_pred_f, f_pred_t, t_pred_f, t_pred_t = [], [], [], []
    t_correctness, f_correctness = 0, 0
    t_tok = model.tokenizer(" TRUE").input_ids[-1]
    f_tok = model.tokenizer(" FALSE").input_ids[-1]
    
    for idx in tqdm(range(len(true_prompts))):
        false_prompt = false_prompts[idx]
        true_prompt = true_prompts[idx]
        if need_question == False:
            false_prompt, label = false_prompt[0], false_prompt[1]
            true_prompt = true_prompt[0]   # true_prompt[1] = false_prompt[1]
            if dataset == 'sp_en_trans':
                if model_name[0].startswith('mistral'):  # Their decoder is slightly different
                    t_tok = model.tokenizer(f"'{label}").input_ids[-1]
                else:
                    t_tok = model.tokenizer(f"{label}").input_ids[-1]
            else:
                t_tok = model.tokenizer(f" {label}").input_ids[-1]

        with model.trace(false_prompt):
            logits = model.lm_head.output[0, -1].save()
        logits = logits.softmax(-1)
        f_pred_t.append(logits[t_tok].item())
        if need_question:
            f_pred_f.append(logits[f_tok].item())
        else:
            logits[t_tok] = 0
            f_pred_f.append(logits.max().item())
        if f_pred_f[-1] > f_pred_t[-1]:
            f_correctness += 1

        with model.trace(true_prompt):
            logits = model.lm_head.output[0, -1].save()
        logits = logits.softmax(-1)
        # max_token_id = logits.argmax(dim=-1).item()
        # output_token = model.tokenizer.decode([max_token_id])
        # print(true_prompt)
        # print(output_token, label, logits[max_token_id], logits[t_tok], max_token_id, t_tok)
        # exit()

        t_pred_t.append(logits[t_tok].item())
        if need_question:
            t_pred_f.append(logits[f_tok].item())
        else:
            logits[t_tok] = 0
            t_pred_f.append(logits.max().item())
        if t_pred_t[-1] > t_pred_f[-1]:
            t_correctness += 1
    
    print(f'F pred to be F: {100 * sum(f_pred_f) / len(f_pred_f):.2f}%, F pred to be T: {100 * sum(f_pred_t) / len(f_pred_t):.2f}%, '
        f'T pred to be F: {100 * sum(t_pred_f) / len(t_pred_f):.2f}%, T pred to be T: {100 * sum(t_pred_t) / len(t_pred_t):.2f}%, '
        f'F correctness: {100 * f_correctness / len(f_pred_f):.2f}%, T correctness: {100 * t_correctness / len(t_pred_t):.2f}%')


def find_dataset_tokens(model_name, dataset, classes, need_question):
    if need_question:
        if dataset == 'cities':   # 15 tokens
            base_tokens =  ["The", "city", "of", "[s1]", "[s2]", "[s3]", "is", "in", "[o1]", "."]
        elif dataset == 'neg_cities':  # 16 tokens
            base_tokens = ["The", "city", "of", "[s1]", "[s2]", "[s3]", "is", "not", "in", "[o1]", "."]
        elif dataset == 'larger_than':   # 14 tokens
            if model_name in ['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft']:
                base_tokens =  ["[s1]", '[s2]', '[s3]', 'is', 'larger', 'than', '[o1]', '[o2]', '.']
            elif model_name in ['llama-2-13b', 'llama-2-13b-instruct']:
                base_tokens = ["[s1]", '[s2]', '[s3]', '[s4]', '[s5]', 'is', 'larger', 'than', '[o1]', '[o2]', '[o3]', '[o4]', '.']
            elif model_name in ['mistral-7B', 'mistral-7B-instruct', 'mistral-7B-SFT']:
                base_tokens = ["[s1]", '[s2]', '[s3]', '[s4]', 'is', 'larger', 'than', '[o1]', '[o2]', '[o3]', '.']
        elif dataset == 'smaller_than':  # 14 tokens
            if model_name in ['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft']:
                base_tokens = ["[s1]", '[s2]', '[s3]', 'is', 'smaller', 'than', '[o1]', '[o2]', '.']
            elif model_name in ['llama-2-13b', 'llama-2-13b-instruct']:
                base_tokens = ["[s1]", '[s2]', '[s3]', '[s4]', '[s5]', 'is', 'smaller', 'than', '[o1]', '[o2]', '[o3]', '[o4]', '.']
            elif model_name in ['mistral-7B', 'mistral-7B-instruct', 'mistral-7B-SFT']:
                base_tokens = ["[s1]", '[s2]', '[s3]', '[s4]', 'is', 'smaller', 'than', '[o1]', '[o2]', '[o3]', '[o4]', '.']
        elif dataset == 'sp_en_trans':   # 16 tokens
            base_tokens = ["The", "Spanish", "word", "'", "[s1]", "[s2]", "' ", "means", "'  ", "[o1]", "'.",]
        elif dataset == 'neg_sp_en_trans':  # 18 tokens
            base_tokens = ["The", "Spanish", "word", "'", "[s1]", "[s2]", "' ", "does", "not", "mean", "'  ", "[o1]", "'."]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if classes in ["true_false", "true_false_noise"]:
            return base_tokens + ["This", "statement", "is ", ":"]
        else:
            raise ValueError(f"Unknown class: {classes}")
    
    else:
        if dataset == 'cities':
            base_tokens =  ["The", "city", "of", "[s1]", "[s2]", "[s3]", "is", "in"]
        elif dataset == 'sp_en_trans':   # 16 token
            base_tokens = ["The", "Spanish", "word", "'", "[s1]", "[s2]", "' ", "means", "'  "]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        return base_tokens


# Patching experiments: generate the patching activation matrix
def patching_one_model(model_name, dataset, device, classes, need_question):
    false_prompts, true_prompts = load_data(model_name, dataset, classes, need_question)
    model = load_model(model_name, device=device)
    layers = model.model.layers
    outputs = []

    t_tok = model.tokenizer(" TRUE").input_ids[-1]
    f_tok = model.tokenizer(" FALSE").input_ids[-1]

    for idx in tqdm(range(len(true_prompts))):
        false_prompt = false_prompts[idx]
        true_prompt = true_prompts[idx]
        if need_question == False:
            false_prompt = false_prompt[0]
            true_prompt = true_prompt[0]
        false_prompt_toks = model.tokenizer(false_prompt).input_ids
        true_prompt_toks = model.tokenizer(true_prompt).input_ids
        try:
            assert len(false_prompt_toks) == len(true_prompt_toks)
        except:
            print(false_prompt)
            print(true_prompt)
            print(len(false_prompt_toks), len(true_prompt_toks))
            exit()

        if args.dataset == 'tulu_extracted':  # number of tokens after the few-shot examples
            if model_name.startswith('mistral'):
                n_toks = len(false_prompt_toks) - 70
            elif model_name.startswith('llama-2'):
                n_toks = len(false_prompt_toks) - 69
            else:
                assert model_name.startswith('llama-3.1-8b')
                n_toks = len(false_prompt_toks) - 65
        else:
            n_toks = len(find_dataset_tokens(model_name, dataset, classes, need_question))

        if classes == 'true_false':
            true_acts = []
            with model.trace(true_prompt):
                for layer in model.model.layers:
                    true_acts.append(layer.output[0].save())
                logits = model.lm_head.output[0, -1].save()
            true_acts = [act.value for act in true_acts]   # the length is 32, because the model has 32 layers
            logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]

            if need_question == False:
                t_tok = logits.argmax(dim=-1).item()
                with model.trace(false_prompt):
                    logits = model.lm_head.output[0, -1].save()
                f_tok = logits.argmax(dim=-1).item()

            for tok_idx in range(1, n_toks + 1):
                for layer_idx, layer in enumerate(model.model.layers):
                    with model.trace(false_prompt):
                        layer.output[0][0, -tok_idx, :] = true_acts[layer_idx][0, -tok_idx, :]
                        logits = model.lm_head.output
                        logit_diff = logits[0, -1, t_tok] - logits[0, -1, f_tok]
                        logit_diff = logit_diff.save()
                    lf = logit_diff.detach().cpu()
                    logit_diffs[tok_idx - 1][layer_idx] = lf.item()
            
            output = {
                'false_prompt': false_prompt,
                'true_prompt': true_prompt,
                'logit_diffs': logit_diffs
            }
            outputs.append(output)
        
        elif classes == 'true_false_noise':   # Use random noise to do patching instead of true-false pairs
            tokens = find_dataset_tokens(model_name, dataset, classes)
            s_pattern = re.compile(r"\[s\d+\]")
            s_indices = [i for i, item in enumerate(tokens) if s_pattern.fullmatch(item)]
            s_indices = [-(len(tokens) - i) for i in s_indices]

            for (clean_prompt, label) in [(false_prompt, 0), (true_prompt, 1)]:
                clean_acts = []
                with model.trace(clean_prompt):
                    for layer in model.model.layers:
                        clean_acts.append(layer.output[0].save())
                    subject_token_embed = model.model.embed_tokens.output[0, s_indices].save()
                clean_acts = [act.value for act in clean_acts]   # the length is 32, because the model has 32 layers
                noise_std = 0.05 * subject_token_embed.abs()
                noise = torch.randn_like(subject_token_embed) * noise_std
                noisy_subject_token_embed = subject_token_embed + noise
                logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]

                for tok_idx in range(1, n_toks + 1):
                    for layer_idx, layer in enumerate(model.model.layers):
                        with model.trace(clean_prompt):
                            model.model.embed_tokens.output[0, s_indices] = noisy_subject_token_embed  # construct noisy input
                            layer.output[0][0, -tok_idx, :] = clean_acts[layer_idx][0, -tok_idx, :]
                            logits = model.lm_head.output
                            logit_diff = logits[0, -1, t_tok] - logits[0, -1, f_tok]
                            logit_diff = logit_diff.save()
                        lf = logit_diff.detach().cpu()
                        logit_diffs[tok_idx - 1][layer_idx] = lf.item()
                
                output = {
                    'label': label,
                    'prompt': clean_prompt,
                    'logit_diffs': logit_diffs
                }
                outputs.append(output)
    
    print(f'Saving {len(outputs)} outputs')
    if need_question == True:
        file_path = 'experimental_outputs/patching_results.json'
    else:
        file_path = 'experimental_outputs/patching_results_no_question.json'

    # Ensure the file exists before entering the loop
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f, indent=4)  # Create an empty JSON object
    while True:
        try:
            with open(file_path, 'r') as f:
                json_outputs = json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(5)
            continue  # Retry reading the file
        json_outputs[f'{model_name}_{dataset}_{classes}'] = outputs
        try:
            with open(file_path, 'w') as f:
                json.dump(json_outputs, f, indent=4)
            break  # Successfully written, exit loop
        except OSError:
            time.sleep(5)


# Patching experiments: generate the patching activation matrix by patching from one model to another
# Patch the true statements' activations from model 1 to false statements' activations in model2
def patching_two_models(model_names, dataset, device, classes, need_question):
    if 'mistral-7B-SFT' in model_names:
        false_prompts, true_prompts = load_data('mistral-7B-SFT', dataset, classes, need_question)
    else:
        false_prompts, true_prompts = load_data(model_names[0], dataset, classes, need_question)
    model1 = load_model(model_names[0], device=device)
    model2 = load_model(model_names[1], device=device)
    layers = model1.model.layers
    outputs = []

    assert classes == 'true_false'
    t_tok = model1.tokenizer(" TRUE").input_ids[-1]
    f_tok = model1.tokenizer(" FALSE").input_ids[-1]

    for idx in tqdm(range(len(true_prompts))):
        false_prompt = false_prompts[idx]
        true_prompt = true_prompts[idx]
        if need_question == False:
            false_prompt = false_prompt[0]
            true_prompt = true_prompt[0]
        false_prompt_toks = model1.tokenizer(false_prompt).input_ids
        true_prompt_toks = model1.tokenizer(true_prompt).input_ids
        assert len(false_prompt_toks) == len(true_prompt_toks)

        if args.dataset == 'tulu_extracted':  # number of tokens after the few-shot examples
            if model_names[0].startswith('mistral-7B'):
                n_toks = len(false_prompt_toks) - 70
            else:
                assert model_names[0].startswith('llama-3.1-8b')
                n_toks = len(false_prompt_toks) - 64
        else:
            n_toks = len(find_dataset_tokens(model_names[0], dataset, classes, need_question))
        
        true_acts = []
        with model1.trace(true_prompt):
            for layer in model1.model.layers:
                true_acts.append(layer.output[0].save())
            logits = model1.lm_head.output[0, -1].save()
        true_acts = [act.value for act in true_acts]   # the length is 32, because the model has 32 layers
        logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]

        if need_question == False:
            t_tok = logits.argmax(dim=-1).item()
            with model2.trace(false_prompt):
                logits = model2.lm_head.output[0, -1].save()
            f_tok = logits.argmax(dim=-1).item()

        for tok_idx in range(1, n_toks + 1):
            for layer_idx, layer in enumerate(model2.model.layers):
                with model2.trace(false_prompt):
                    layer.output[0][0, -tok_idx, :] = true_acts[layer_idx][0, -tok_idx, :]
                    logits = model2.lm_head.output
                    logit_diff = logits[0, -1, t_tok] - logits[0, -1, f_tok]
                    logit_diff = logit_diff.save()
                lf = logit_diff.detach().cpu()
                logit_diffs[tok_idx - 1][layer_idx] = lf.item()
        
        output = {
            'false_prompt': false_prompt,
            'true_prompt': true_prompt,
            'logit_diffs': logit_diffs
        }
        outputs.append(output)
    
    print(f'Saving {len(outputs)} outputs')
    if need_question == True:
        file_path = 'experimental_outputs/patching_results.json'
    else:
        file_path = 'experimental_outputs/patching_results_no_question.json'

    # Ensure the file exists before entering the loop
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump({}, f, indent=4)  # Create an empty JSON object
    while True:
        try:
            with open(file_path, 'r') as f:
                json_outputs = json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(5)
            continue  # Retry reading the file
        json_outputs[f'{model_names[0]}_{model_names[1]}_{dataset}_{classes}'] = outputs
        try:
            with open(file_path, 'w') as f:
                json.dump(json_outputs, f, indent=4)
            break  # Successfully written, exit loop
        except OSError:
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=['llama-2-13b'],
        nargs='+', choices=['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft',
        'llama-2-13b', 'llama-2-13b-instruct',
        'mistral-7B', 'mistral-7B-SFT', 'mistral-7B-instruct',
        'deepseek-V2-lite', 'deepseek-V2-lite-chat'])
    parser.add_argument('--dataset', type=str, default='cities')
    parser.add_argument('--classes', type=str, default='true_false',
        choices=['true_false', 'true_false_noise'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--need_question', type=bool, default=True,
        help='Whether to add "This statement is:" at the end of the prompt.')
    args = parser.parse_args()
    print(args)

    # check_prob_one_model(args.model, args.dataset, args.device, args.classes, args.need_question)

    if len(args.model) == 1:
        patching_one_model(args.model[0], args.dataset, args.device, args.classes, args.need_question)
    else:
        assert len(args.model) == 2
        patching_two_models(args.model, args.dataset, args.device, args.classes, args.need_question)
