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


# prompt for cities
false_prompt = "The city of Tokyo is in Japan. This statement is: TRUE\nThe city of Hanoi is in Poland. This statement is: FALSE\nThe city of Chicago is in Canada. This statement is:"
true_prompt = "The city of Tokyo is in Japan. This statement is: TRUE\nThe city of Hanoi is in Poland. This statement is: FALSE\nThe city of Toronto is in Canada. This statement is:"

# prompt for larger_than
# false_prompt = "Fifty-one is larger than seventy-five. This statement is: FALSE\nEighty-two is larger than sixty-four. This statement is: TRUE\nEighty-one is larger than eighty-four. This statement is:"
# true_prompt = "Fifty-one is larger than seventy-five. This statement is: FALSE\nEighty-two is larger than sixty-four. This statement is: TRUE\nEighty-six is larger than eighty-four. This statement is:"

# prompt for sp_en_trans
'''
false_prompt = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'diccionario' means 'dictionary'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
The Spanish word 'con' means 'one'. This statement is:"""
true_prompt = """\
The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
The Spanish word 'escribir' means 'to write'. This statement is: TRUE
The Spanish word 'diccionario' means 'dictionary'. This statement is: TRUE
The Spanish word 'gato' means 'cat'. This statement is: TRUE
The Spanish word 'aire' means 'silver'. This statement is: FALSE
The Spanish word 'uno' means 'one'. This statement is:"""

false_prompt = """\
The city of Barnaul is in South Africa. This statement is: FALSE
The city of Leon de los Aldama is in China. This statement is: FALSE
The city of Kyiv is in Russia. This statement is: FALSE
The city of Liupanshui is in the Philippines. This statement is: FALSE
The city of Ahvaz is in China. This statement is: FALSE
The city of Kyiv is in China. This statement is: """
true_prompt = """\
The city of Barnaul is in South Africa. This statement is: FALSE
The city of Leon de los Aldama is in China. This statement is: FALSE
The city of Kyiv is in Russia. This statement is: FALSE
The city of Liupanshui is in the Philippines. This statement is: FALSE
The city of Ahvaz is in China. This statement is: FALSE
The city of Anqing is in China. This statement is: """
    
false_prompt = """\
"Ninety-two is larger than fifty-five. This statement is: TRUE
Seventy-eight is larger than ninety-five. This statement is: FALSE
Seventy-one is larger than ninety-one. This statement is: FALSE
Sixty-six is larger than ninety-nine. This statement is: FALSE
Seventy-four is larger than seventy-three. This statement is: TRUE
Fifty-two is larger than fifty-four This statement is: """
true_prompt = """\
"Ninety-two is larger than fifty-five. This statement is: TRUE
Seventy-eight is larger than ninety-five. This statement is: FALSE
Seventy-one is larger than ninety-one. This statement is: FALSE
Sixty-six is larger than ninety-nine. This statement is: FALSE
Seventy-four is larger than seventy-three. This statement is: TRUE
Fifty-six is larger than fifty-four This statement is: """
'''

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For complete reproducibility, especially with convolutional operations, add:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(model_name, dataset, classes):
    if len(dataset) == 0:
        false_prompts = [false_prompt]
        true_prompts = [true_prompt]
    
    else:
        if model_name in ['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft']:
            model_family = 'llama3'
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

        # Construct the few-shot examples
        if classes in ['true_false', 'true_false_noise']:
            example_list = [f"{example} This statement is: TRUE\n" for example in example_true_list]
            example_list.extend([f"{example} This statement is: FALSE\n" for example in example_false_list])
        else:
            raise ValueError(f"Class {classes} not recognized")

        final_true_prompts = []
        final_false_prompts = []
        for i in range(len(true_prompts)):
            random.shuffle(example_list)  # randomly perturb the order of the examples
            if classes in ['true_false', 'true_false_noise']:
                final_true_prompts.append(''.join(example_list) + true_prompts[i] + " This statement is:")
                final_false_prompts.append(''.join(example_list) + false_prompts[i] + " This statement is:")
            else:
                raise ValueError(f"Class {classes} not recognized")
        true_prompts = final_true_prompts
        false_prompts = final_false_prompts
    
    if model_name == 'mistral-7B-SFT':   # Fix its tokenizer's bug
        true_prompts = ['<s> ' + prompt for prompt in true_prompts]
        false_prompts = ['<s> ' + prompt for prompt in false_prompts]
    return false_prompts, true_prompts


# Calculate the model’s average likelihood of the two classes.
def check_prob_one_model(model_name, dataset, device, classes):
    assert len(model_name) == 1
    false_prompts, true_prompts = load_data(model_name[0], dataset, classes)
    model = load_model(model_name[0], device=device)
    f_pred_f, f_pred_t, t_pred_f, t_pred_t = [], [], [], []
    t_correctness, f_correctness = 0, 0
    t_tok = model.tokenizer(" TRUE").input_ids[-1]
    f_tok = model.tokenizer(" FALSE").input_ids[-1]
    
    for idx in tqdm(range(len(true_prompts))):
        false_prompt = false_prompts[idx]
        true_prompt = true_prompts[idx]
        with model.trace(false_prompt):
            logits = model.lm_head.output[0, -1].save()
        logits = logits.softmax(-1)
        f_pred_f.append(logits[f_tok].item())
        f_pred_t.append(logits[t_tok].item())
        if f_pred_f[-1] > f_pred_t[-1]:
            f_correctness += 1
            tf_token = 'FALSE'
        else:
            tf_token = 'TRUE'

        with model.trace(true_prompt):
            logits = model.lm_head.output[0, -1].save()
        logits = logits.softmax(-1)
        t_pred_f.append(logits[f_tok].item())
        t_pred_t.append(logits[t_tok].item())
        if t_pred_t[-1] > t_pred_f[-1]:
            t_correctness += 1
            tf_token = 'TRUE'
        else:
            tf_token = 'FALSE'
    
    print(f'F pred to be F: {100 * sum(f_pred_f) / len(f_pred_f):.2f}%, F pred to be T: {100 * sum(f_pred_t) / len(f_pred_t):.2f}%, '
        f'T pred to be F: {100 * sum(t_pred_f) / len(t_pred_f):.2f}%, T pred to be T: {100 * sum(t_pred_t) / len(t_pred_t):.2f}%, '
        f'F correctness: {100 * f_correctness / len(f_pred_f):.2f}%, T correctness: {100 * t_correctness / len(t_pred_t):.2f}%')


def find_dataset_tokens(model_name, dataset, classes):
    if dataset == 'cities':   # 15 tokens
        base_tokens =  ["The", "city", "of", "[s1]", "[s2]", "[s3]", "is", "in", "[o1]", "."]
    elif dataset == 'neg_cities':  # 16 tokens
        base_tokens = ["The", "city", "of", "[s1]", "[s2]", "[s3]", "is", "not", "in", "[o1]", "."]
    elif dataset == 'larger_than':   # 14 tokens
        if model_name in ['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft']:
            base_tokens =  ["[s1]", '[s2]', '[s3]', 'is', 'larger', 'than', '[o1]', '[o2]', '.']
        elif model_name in ['mistral-7B', 'mistral-7B-instruct', 'mistral-7B-SFT']:
            base_tokens = ["[s1]", '[s2]', '[s3]', '[s4]', 'is', 'larger', 'than', '[o1]', '[o2]', '[o3]', '.']
    elif dataset == 'smaller_than':  # 14 tokens
        if model_name in ['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft']:
            base_tokens = ["[s1]", '[s2]', '[s3]', 'is', 'smaller', 'than', '[o1]', '[o2]', '.']
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


# Patching experiments: generate the patching activation matrix
def patching_one_model(model_name, dataset, device, classes):
    false_prompts, true_prompts = load_data(model_name, dataset, classes)
    model = load_model(model_name, device=device)
    layers = model.model.layers
    outputs = []

    t_tok = model.tokenizer(" TRUE").input_ids[-1]
    f_tok = model.tokenizer(" FALSE").input_ids[-1]

    for idx in tqdm(range(len(true_prompts))):
        false_prompt = false_prompts[idx]
        true_prompt = true_prompts[idx]
        false_prompt_toks = model.tokenizer(false_prompt).input_ids
        true_prompt_toks = model.tokenizer(true_prompt).input_ids
        assert len(false_prompt_toks) == len(true_prompt_toks)

        if args.dataset == 'tulu_extracted':  # number of tokens after the few-shot examples
            if model_name.startswith('mistral'):
                n_toks = len(false_prompt_toks) - 70
            else:
                n_toks = len(false_prompt_toks) - 65
        else:
            n_toks = len(find_dataset_tokens(model_name, dataset, classes))

        if classes == 'true_false':
            true_acts = []
            with model.trace(true_prompt):
                for layer in model.model.layers:
                    true_acts.append(layer.output[0].save())
            true_acts = [act.value for act in true_acts]   # the length is 32, because the model has 32 layers
            logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]

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
    file_path = 'experimental_outputs/patching_results.json'

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
def patching_two_models(model_names, dataset, device, classes):
    if 'mistral-7B-SFT' in model_names:
        false_prompts, true_prompts = load_data('mistral-7B-SFT', dataset, classes)
    else:
        false_prompts, true_prompts = load_data(model_names[0], dataset, classes)
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
            n_toks = len(find_dataset_tokens(model_names[0], dataset, classes))
        
        true_acts = []
        with model1.trace(true_prompt):
            for layer in model1.model.layers:
                true_acts.append(layer.output[0].save())
        true_acts = [act.value for act in true_acts]   # the length is 32, because the model has 32 layers
        logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]

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
    file_path = 'experimental_outputs/patching_results.json'

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
    parser.add_argument('--model', type=str, default=['llama-3.1-8b-instruct'],
        nargs='+', choices=['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft',
        'mistral-7B', 'mistral-7B-SFT', 'mistral-7B-instruct',
        'deepseek-V2-lite', 'deepseek-V2-lite-chat'])
    parser.add_argument('--dataset', type=str, default='cities')
    parser.add_argument('--classes', type=str, default='true_false',
        choices=['true_false', 'true_false_noise'])
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    print(args)

    # check_prob_one_model(args.model, args.dataset, args.device, args.classes)

    if len(args.model) == 1:
        patching_one_model(args.model[0], args.dataset, args.device, args.classes)
    else:
        assert len(args.model) == 2
        patching_two_models(args.model, args.dataset, args.device, args.classes)
