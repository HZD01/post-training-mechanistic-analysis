import plotly.express as px
import json
from nnsight import LanguageModel
import json
import numpy as np
import torch
import os
import argparse
from scipy.stats import spearmanr, pearsonr
import re
import matplotlib.pyplot as plt
import plotly.io as pio
pio.kaleido.scope.mathjax = None

from patching import find_dataset_tokens
from generate_acts import load_model


def normalize(mean_array, classes):
    values = mean_array.flatten()
    # Draw a histogram of the values
    # plt.hist(values, bins=20, edgecolor='black', alpha=0.7)  # Adjust 'bins' as needed
    # xlabel = "log P(T)/P(F)" if classes == 'true_false' else "log P(C)/P(U)"
    # plt.xlabel(xlabel)
    # plt.ylabel("Frequency")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig('test.png', bbox_inches='tight')
    # exit()

    bins = np.linspace(np.min(values) - 1e-5, np.max(values) + 1e-5, 21)  # Create 20 bins
    bin_indices = np.digitize(values, bins)  # Get the bin indices for each value
    # New values to assign for each bin: Bins 1-10: 0, Bin 11: 0.1, Bin 12: 0.2, ..., Bin 20: 1
    bin_values = np.concatenate((np.zeros(10), np.linspace(0.1, 1.0, 10)))
    updated_values = bin_values[bin_indices - 1]  # `-1` because bin indices are 1-based
    mean_array = updated_values.reshape(mean_array.shape)
    return mean_array


def draw_figure(mean_array, tokens, colorbar_title, title, save_name):
    soft_diverging_scale = [
        (0.0,  'rgb(0, 0, 255)'),  # pastel blue
        (0.5,  'rgb(245, 245, 245)'),  # near-white in the center
        (1.0,  'rgb(255, 0, 0)')   # pastel red
    ]
    if len(tokens) == 0:
        tokens = [str(i) for i in range(mean_array.shape[1])]
    assert mean_array.shape[1] == len(tokens)
    title = title.replace("llama", "Llama").replace("mistral", "Mistral").replace("8b", "8B")
    title = title.replace("instruct", "Instruct").replace("sft", "SFT")

    fig = px.imshow(
        mean_array,
        x=tokens,
        labels=dict(y="Layer", color=f'{colorbar_title}<br><br>'),
        color_continuous_scale=soft_diverging_scale,
        color_continuous_midpoint=0,
        origin='upper',  # so top row = first layer in 'layers' list
        title=title,
        range_color=[-1, 1]
    )

    if len(tokens) > 16:
        base_font_size = 21  # A larger font size will result in incomplete xtick display
    else:
        base_font_size = 23
    fig.update_layout(  # Update font sizes
        title_font_size=base_font_size + 2,
        font=dict(size=base_font_size),  # general font size
        coloraxis_colorbar=dict(title_font_size=base_font_size + 2, tickfont_size=base_font_size),
    )
    fig.update_xaxes(tickfont_size=base_font_size)
    fig.update_yaxes(title_font_size=base_font_size + 2, tickfont_size=base_font_size)
    fig.write_image(f"experimental_outputs/patching_png/{save_name}.png")
    fig.write_image(f"experimental_outputs/patching_pdf/{save_name}.pdf")


def get_key(model_names, dataset, classes):
    if len(model_names) == 2:
        return f"{model_names[0]}_{model_names[1]}_{dataset}_{classes}"
    else:
        assert len(model_names) == 1
        return f"{model_names[0]}_{dataset}_{classes}"


def visualize_results(model_names, dataset, classes="true_false"):
    # For the given model_name and dataset, load all the patching results, average them and visualize them.
    with open('experimental_outputs/patching_results.json', 'r') as f:
        all_results = json.load(f)
    key = get_key(model_names, dataset, classes)
    results_list = all_results[key]
    print(f"Found {len(results_list)} outputs for {key}.")
    n_toks = len(results_list[0]['logit_diffs'])   # Token number
    for result in results_list:
        assert len(result['logit_diffs']) == n_toks
    n_layers = len(results_list[0]['logit_diffs'][0])

    if classes == 'true_false':
        sum_array = np.zeros((n_layers, n_toks), dtype=np.float32)  # Accumulator to sum over all prompts
    else:
        raise ValueError(f"Invalid class: {classes}")

    for result in results_list:
        logit_diffs = result['logit_diffs']  # shape [n_toks, n_layers]
        # shape of new logit_diffs: [n_layers, n_toks]
        logit_diffs = [[logit_diffs[i][j] for i in range(0, len(logit_diffs))[::-1]] for j in range(len(logit_diffs[0]))]
        sum_array += logit_diffs
    
    tokens = find_dataset_tokens(model_names[0], dataset, classes)
    assert len(tokens) == n_toks
    if len(model_names) == 1:
        model_title = model_names[0]
    else:
        model_title = f"{model_names[0]}_to_{model_names[1]}"

    mean_array = sum_array / len(results_list)
    mean_array = normalize(mean_array, classes)
    draw_figure(mean_array, tokens, "log P(T)/P(F)", f"{model_title} {dataset}",
        f"one_result/{model_title}_{dataset}")


def compare_two_results(model_names1, model_names2, dataset, classes):
    # Compare the two models' patching results for the given dataset
    key1 = get_key(model_names1, dataset, classes)
    key2 = get_key(model_names2, dataset, classes)

    with open('experimental_outputs/patching_results.json', 'r') as f:  # Load patching results
        all_data = json.load(f)
    results_list1 = all_data[key1]
    results_list2 = all_data[key2]
    print(f'Found {len(results_list1)} results for {key1} and {len(results_list2)} results for {key2}.')

    n_toks = len(results_list1[0]['logit_diffs'])   # Token number
    for result in results_list1:
        assert len(result['logit_diffs']) == n_toks
    for result in results_list2:
        assert len(result['logit_diffs']) == n_toks
    n_layers = len(results_list2[0]['logit_diffs'][0])

    sum_array1 = np.zeros((n_layers, n_toks), dtype=np.float32)  # Accumulator to sum over all prompts
    sum_array2 = np.zeros((n_layers, n_toks), dtype=np.float32)  # Accumulator to sum over all prompts

    for result in results_list1:
        logit_diffs = result['logit_diffs']  # shape [n_toks, n_layers]
        logit_diffs = [[logit_diffs[i][j] for i in range(0, len(logit_diffs))[::-1]] for j in range(len(logit_diffs[0]))]
        sum_array1 += logit_diffs

    for result in results_list2:
        logit_diffs = result['logit_diffs']  # shape [n_toks, n_layers]
        logit_diffs = [[logit_diffs[i][j] for i in range(0, len(logit_diffs))[::-1]] for j in range(len(logit_diffs[0]))]
        sum_array2 += logit_diffs

    mean_array1 = sum_array1 / len(results_list1)
    mean_array2 = sum_array2 / len(results_list2)
    mean_array1 = normalize(mean_array1, classes)
    mean_array2 = normalize(mean_array2, classes)
    diff_array = mean_array1 - mean_array2

    tokens = find_dataset_tokens(model_names1[0], dataset, classes)
    assert len(tokens) == n_toks
    s_pattern = re.compile(r"\[s\d+\]")
    o_pattern = re.compile(r"\[o\d+\]")
    # Find indices of elements matching the patterns
    knowledge_indices = [i for i, item in enumerate(tokens) if s_pattern.fullmatch(item) or o_pattern.fullmatch(item)]
    diff_array_knowledge = diff_array[:, knowledge_indices]
    period_indices = [i for i, item in enumerate(tokens) if item == '.' or item  == "'."]
    assert len(period_indices) == 1
    diff_array_period = diff_array[:, period_indices[0]:period_indices[0] + 1]
    print(f'Maximum value of the difference matrix: {np.abs(diff_array).max():.1f}, '
        f'knowledge part: {np.abs(diff_array_knowledge).max():.1f}')

    pearson_corr, _ = pearsonr(mean_array1.flatten(), mean_array2.flatten())
    print(f'Pearson correlation of the two matrices: {pearson_corr:.4f}')

    # print()
    # shuffled_array2 = np.random.permutation(mean_array2)  # Shuffle the rows
    # shuffled_array2 = np.random.permutation(shuffled_array2.T).T  # Shuffle the columns
    # shuffled_diff_array = mean_array1 - shuffled_array2
    # print(f'Maximum value of the difference matrix (shuffled): {np.abs(shuffled_diff_array).max()}')
    # relative_diff = np.abs(shuffled_diff_array) / np.maximum(np.abs(mean_array1), np.abs(shuffled_array2))
    # relative_diff = np.nan_to_num(relative_diff)  # Change the nan values to 0
    # print(f'Relative mean difference (shuffled): {relative_diff.mean()}')
    # pearson_corr, _ = pearsonr(mean_array1.flatten(), shuffled_array2.flatten())
    # spearman_corr, _ = spearmanr(mean_array1.flatten(), shuffled_array2.flatten())
    # print(f'Pearson correlation of the two matrices (shuffled): {pearson_corr}; Spearman correlation (shuffled): {spearman_corr}')

    draw_figure(diff_array, tokens, "log P(T)/P(F)", f"{'_to_'.join(model_names1)} - {'_to_'.join(model_names2)}  {dataset}",
        f"compare_two/{'_to_'.join(model_names1)}_{'_to_'.join(model_names2)}_{dataset}")


def visualize_results_tulu(model_names, dataset, classes):
    # For the Tulu dataset, the false and true prompts can have different lengths,
    # and the tokens that differ between the two prompts vary by pair.
    # This function detects the differing token positions as the knowledge-storage positions.
    with open('experimental_outputs/patching_results.json', 'r') as f:
        all_results = json.load(f)
    key = get_key(model_names, dataset, classes)
    model = load_model(model_names[0], device='cuda')
    results_list = all_results[key]
    print(f"Found {len(results_list)} outputs for {key}.")
    n_layers = len(results_list[0]['logit_diffs'][0])

    assert classes == 'true_false'
    sum_array_knowledge = np.zeros(n_layers, dtype=np.float32)  # Accumulator to sum over all prompts
    sum_array_last_token = np.zeros(n_layers, dtype=np.float32)
    sum_array_others = np.zeros(n_layers, dtype=np.float32)
    max_knowledge_tokens = 0

    for result in results_list:
        logit_diffs = result['logit_diffs']  # shape [n_toks, n_layers]
        # shape of new logit_diffs: [n_layers, n_toks]
        logit_diffs = [[logit_diffs[i][j] for i in range(0, len(logit_diffs))[::-1]] for j in range(len(logit_diffs[0]))]
        logit_diffs = np.array(logit_diffs)
        if model_names[0].startswith('llama-3.1-8b'):
            few_shot_example_toks = 64
        else:
            assert model_names[0].startswith('mistral-7B')
            few_shot_example_toks = 70
        false_prompt_toks = model.tokenizer(result['false_prompt']).input_ids[few_shot_example_toks:]
        true_prompt_toks = model.tokenizer(result['true_prompt']).input_ids[few_shot_example_toks:]

        assert len(false_prompt_toks) == len(logit_diffs[0]) and len(true_prompt_toks) == len(logit_diffs[0])
        knowledge_indices = [i for i in range(len(false_prompt_toks)) if false_prompt_toks[i] != true_prompt_toks[i]]
        other_indices = [i for i in range(len(false_prompt_toks)) if i not in knowledge_indices and i != len(false_prompt_toks) - 1]
        assert len(knowledge_indices) > 0
        max_knowledge_tokens = max(max_knowledge_tokens, len(knowledge_indices))
        sum_array_knowledge += np.mean(logit_diffs[:, knowledge_indices], axis=1)
        sum_array_last_token += logit_diffs[:, -1]
        sum_array_others += np.mean(logit_diffs[:, other_indices], axis=1)
    
    print(f'Maximum number of knowledge tokens: {max_knowledge_tokens}')
    mean_array_knowledge = sum_array_knowledge / len(results_list)
    mean_array_last_token = sum_array_last_token / len(results_list)
    mean_array_others = sum_array_others / len(results_list)
    mean_array = np.concatenate([mean_array_knowledge.reshape(-1, 1), mean_array_others.reshape(-1, 1),
        mean_array_last_token.reshape(-1, 1)], axis=1)
    mean_array = normalize(mean_array, classes)
    print(f'Maximum value of the knowledge matrix: {np.abs(mean_array[:, 0]).max():.1f}, '
        f'last token: {np.abs(mean_array[:, 2]).max():.1f} ',
        f'others parts: {np.abs(mean_array[:, 1]).max():.1f}')
    
    tokens = ['knowledge', 'others', 'last_token']
    if len(model_names) == 1:
        model_title = model_names[0]
    else:
        model_title = f"{model_names[0]}_to_{model_names[1]}"
    draw_figure(mean_array, tokens, "log P(T)/P(F)", f"{model_title} {dataset}",
        f"one_result/{model_title}_{dataset}")


def compare_two_results_tulu(model_names1, model_names2, dataset, classes):
    # Compare the two models' patching results for the given dataset
    key1 = get_key(model_names1, dataset, classes)
    key2 = get_key(model_names2, dataset, classes)
    model1 = load_model(model_names1[0], device='cuda')

    with open('experimental_outputs/patching_results.json', 'r') as f:  # Load patching results
        all_data = json.load(f)
    results_list1 = all_data[key1]
    results_list2 = all_data[key2]
    print(f'Found {len(results_list1)} results for {key1} and {len(results_list2)} results for {key2}.')
    n_layers = len(results_list2[0]['logit_diffs'][0])

    assert classes == 'true_false'
    sum_array_knowledge1 = np.zeros(n_layers, dtype=np.float32)  # Accumulator to sum over all prompts
    sum_array_last_token1 = np.zeros(n_layers, dtype=np.float32)
    sum_array_others1 = np.zeros(n_layers, dtype=np.float32)
    sum_array_knowledge2 = np.zeros(n_layers, dtype=np.float32)  # Accumulator to sum over all prompts
    sum_array_last_token2 = np.zeros(n_layers, dtype=np.float32)
    sum_array_others2 = np.zeros(n_layers, dtype=np.float32)

    if model_names1[0].startswith('llama-3.1-8b'):
        few_shot_example_toks = 64
    else:
        assert model_names1[0].startswith('mistral-7B')
        few_shot_example_toks = 70

    for result in results_list1:
        logit_diffs = result['logit_diffs']  # shape [n_toks, n_layers]
        logit_diffs = [[logit_diffs[i][j] for i in range(0, len(logit_diffs))[::-1]] for j in range(len(logit_diffs[0]))]
        logit_diffs = np.array(logit_diffs)
        false_prompt_toks = model1.tokenizer(result['false_prompt']).input_ids[few_shot_example_toks:]
        true_prompt_toks = model1.tokenizer(result['true_prompt']).input_ids[few_shot_example_toks:]
        assert len(false_prompt_toks) == len(logit_diffs[0]) and len(true_prompt_toks) == len(logit_diffs[0])
        knowledge_indices = [i for i in range(len(false_prompt_toks)) if false_prompt_toks[i] != true_prompt_toks[i]]
        other_indices = [i for i in range(len(false_prompt_toks)) if i not in knowledge_indices and i != len(false_prompt_toks) - 1]
        assert len(knowledge_indices) > 0
        sum_array_knowledge1 += np.mean(logit_diffs[:, knowledge_indices], axis=1)
        sum_array_last_token1 += logit_diffs[:, -1]
        sum_array_others1 += np.mean(logit_diffs[:, other_indices], axis=1)

    for result in results_list2:
        logit_diffs = result['logit_diffs']  # shape [n_toks, n_layers]
        logit_diffs = [[logit_diffs[i][j] for i in range(0, len(logit_diffs))[::-1]] for j in range(len(logit_diffs[0]))]
        logit_diffs = np.array(logit_diffs)
        false_prompt_toks = model1.tokenizer(result['false_prompt']).input_ids[few_shot_example_toks:]
        true_prompt_toks = model1.tokenizer(result['true_prompt']).input_ids[few_shot_example_toks:]

        assert len(false_prompt_toks) == len(logit_diffs[0]) and len(true_prompt_toks) == len(logit_diffs[0])
        knowledge_indices = [i for i in range(len(false_prompt_toks)) if false_prompt_toks[i] != true_prompt_toks[i]]
        other_indices = [i for i in range(len(false_prompt_toks)) if i not in knowledge_indices and i != len(false_prompt_toks) - 1]
        assert len(knowledge_indices) > 0
        sum_array_knowledge2 += np.mean(logit_diffs[:, knowledge_indices], axis=1)
        sum_array_last_token2 += logit_diffs[:, -1]
        sum_array_others2 += np.mean(logit_diffs[:, other_indices], axis=1)

    mean_array_knowledge1 = sum_array_knowledge1 / len(results_list1)
    mean_array_last_token1 = sum_array_last_token1 / len(results_list1)
    mean_array_others1 = sum_array_others1 / len(results_list1)
    mean_array1 = np.concatenate([mean_array_knowledge1.reshape(-1, 1), mean_array_others1.reshape(-1, 1),
        mean_array_last_token1.reshape(-1, 1)], axis=1)
    mean_array1 = normalize(mean_array1, classes)
    
    mean_array_knowledge2 = sum_array_knowledge2 / len(results_list2)
    mean_array_last_token2 = sum_array_last_token2 / len(results_list2)
    mean_array_others2 = sum_array_others2 / len(results_list2)
    mean_array2 = np.concatenate([mean_array_knowledge2.reshape(-1, 1), mean_array_others2.reshape(-1, 1),
        mean_array_last_token2.reshape(-1, 1)], axis=1)
    mean_array2 = normalize(mean_array2, classes)
    diff_array = mean_array1 - mean_array2

    print(f'Maximum value of the difference matrix: {np.abs(diff_array).max():.1f}, '
        f'knowledge part: {np.abs(diff_array[:, 0]).max():.1f}')
    pearson_corr, _ = pearsonr(mean_array1.flatten(), mean_array2.flatten())
    print(f'Pearson correlation of the two matrices: {pearson_corr:.4f}')

    tokens = ['knowledge', 'others', 'last_token']
    draw_figure(diff_array, tokens, "log P(T)/P(F)", f"{'_to_'.join(model_names1)} - {'_to_'.join(model_names2)}  {dataset}",
        f"compare_two/{'_to_'.join(model_names1)}_{'_to_'.join(model_names2)}_{dataset}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names1', type=str, nargs='+', default=['llama-3.1-8b-instruct', 'llama-3.1-8b'],
        choices=['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft',
        'mistral-7B', 'mistral-7B-instruct', 'mistral-7B-SFT'],
        help='The first result to visualize or compare.')
    parser.add_argument('--model_names2', type=str, nargs='+', default=['llama-3.1-8b-instruct'],
        choices=['llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-8b-sft',
        'mistral-7B', 'mistral-7B-instruct', 'mistral-7B-SFT'],
        help='The second result to visualize or compare. If only visualize one result, set this one to [].')
    parser.add_argument('--dataset', type=str, default='cities')
    parser.add_argument('--classes', type=str, default='true_false')
    args = parser.parse_args()
    print(args)

    all_model_names = args.model_names1 + args.model_names2
    if all_model_names[0].startswith('llama-3.1-8b'):
        for model_name in all_model_names:
            assert model_name.startswith('llama-3.1-8b')
    elif all_model_names[0].startswith('mistral-7B'):
        for model_name in all_model_names:
            assert model_name.startswith('mistral-7B')
    else:
        raise ValueError(f"Invalid model name: {all_model_names[0]}")

    if args.dataset != 'tulu_extracted':
        if len(args.model_names2) > 0:
            compare_two_results(args.model_names1, args.model_names2, args.dataset, args.classes)
        else:
            visualize_results(args.model_names1, args.dataset, args.classes)
    else:  # For the Tulu dataset, the false and true prompts can have different lengths,
        # and the tokens that differ between the two prompts vary by pair.
        if len(args.model_names2) > 0:
            compare_two_results_tulu(args.model_names1, args.model_names2, args.dataset, args.classes)
        else:
            visualize_results_tulu(args.model_names1, args.dataset, args.classes)
