import os
import sys
import pandas as pd
import plotly.express as px
from utils import *
import plotly.graph_objects as go
import gc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import json
from pathlib import Path

# Model name mapping
MODEL_NAME_MAPPING = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b-instruct": "meta-llama/Llama-2-7b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "mistral-7b-sft": "nyu-dice-lab/Mistral-7B-Base-SFT-Tulu2",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-2-13b": "meta-llama/Llama-2-13B-hf",
    "llama-2-13b-instruct": "meta-llama/Llama-2-13B-chat-hf",

}


# Result file
RESULT_FILE = "neuron_analysis_results.json"

def plot_kde_and_skewness(data, filename="kde_plot.png"):
    if data.empty:
        raise ValueError("The data list is empty.")

    # Compute skewness
    skewness_score = skew(data)

    # Plot KDE
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data, fill=True, color='blue')
    plt.title(f'KDE Plot (Skewness: {skewness_score:.3f})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close to prevent inline display in some environments

    print(f"KDE plot saved as {filename}")
    print(f"Skewness Score: {skewness_score:.3f}")

    return skewness_score

def filter_neurons(df, norm_col="norm", var_col="cos_var", norm_top_percentage=0.25, var_bottom_n=10):
    """
    Filter neurons by:
    1. First selecting top norm_top_percentage% of neurons by norm
    2. Then selecting bottom var_bottom_n neurons by variance from that subset
    """
    if not (0 < norm_top_percentage <= 1):
        raise ValueError("norm_top_percentage must be between 0 and 1.")
    
    if var_bottom_n < 1:
        raise ValueError("var_bottom_n must be at least 1.")

    # Get top percentage of neurons by norm
    norm_threshold = df[norm_col].quantile(1 - norm_top_percentage)
    df_top_norm = df[df[norm_col] > norm_threshold]
    
    # If we have fewer neurons than requested bottom_n, return all of them
    if len(df_top_norm) <= var_bottom_n:
        return df_top_norm
    
    # Get bottom n neurons by variance
    df_filtered = df_top_norm.nsmallest(var_bottom_n, var_col)
    
    # Get the filtered neuron indexes and their ratios
    filtered_neurons = df_filtered['neuron_index'].tolist()
    filtered_ratios = df_filtered['norm_log_var_ratio'].tolist()
    
    # Create a dictionary mapping neuron_index to ratio
    result = {int(idx): float(ratio) for idx, ratio in zip(filtered_neurons, filtered_ratios)}
    
    return result

def compute_iou(a, b):
    """Compute IoU of two sets of neurons (using keys from dictionaries)"""
    set_a = set(a.keys())
    set_b = set(b.keys())
    intersection = set_a & set_b
    union = set_a | set_b
    out = len(intersection) / len(union) if len(union) > 0 else 0.0
    return out

def process_model(model_name):
    """Process a model and return the DataFrame with neuron data"""
    # Map short name to full HF model path
    hf_model_name = MODEL_NAME_MAPPING.get(model_name, model_name)
    print(f"Processing model: {hf_model_name}")
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:   
        device = 'cpu'
    transformers_cache_dir = None
    
    # Load model
    model, _ = load_model_from_tl_name(hf_model_name, device, transformers_cache_dir, hf_token=None)
    model = model.to(device)
    
    # Extract neuron data
    last_layer_neurons = model.W_out[-1]
    norm = torch.norm(last_layer_neurons, dim=1)
    unemb = model.W_U
    
    # Clean up GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    normalized_composition = last_layer_neurons / last_layer_neurons.norm(dim=1).unsqueeze(1) @ unemb / unemb.norm(dim=0)
    cos_var = normalized_composition.var(dim=1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'norm': norm.cpu().detach(),
        'cos_var': cos_var.cpu().detach()
    })
    df['neuron_index'] = df.index
    
    # Calculate the ratio of norm to log(cos_var)
    # Handle potential zeros or negative values in cos_var
    df['norm_log_var_ratio'] = df.apply(
        lambda row: float(row['norm'] / np.log(max(1e-10, row['cos_var']))), 
        axis=1
    )
    
    return df

def load_existing_results():
    """Load existing results from JSON file if it exists"""
    if Path(RESULT_FILE).exists():
        try:
            with open(RESULT_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing {RESULT_FILE}, creating new file")
    return {}

def save_results(results):
    """Save results to JSON file"""
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULT_FILE}")

def main(model1, model2, norm_top_percentages=[0.25], var_bottom_ns=[10]):
    # Load existing results if available
    all_results = load_existing_results()
    
    # Create a unique key for this analysis run
    analysis_key = f"{model1}_vs_{model2}_{'-'.join(map(str, norm_top_percentages))}_{'-'.join(map(str, var_bottom_ns))}"
    
    # Process each model and store results
    model_results = {}
    for model_name in [model1, model2]:
        model_results[model_name] = process_model(model_name)
    
    # Create or update the results structure
    if analysis_key not in all_results:
        all_results[analysis_key] = {
            "models": [model1, model2],
            "parameters": {
                "norm_top_percentages": norm_top_percentages,
                "var_bottom_ns": var_bottom_ns
            },
            "results": {}
        }
    
    # Run analysis for each parameter combination
    for norm_top_percentage in norm_top_percentages:
        for var_bottom_n in var_bottom_ns:
            param_key = f"norm_top_{norm_top_percentage}_var_bottom_{var_bottom_n}"
            print(f"\nAnalysis with norm_top_percentage={norm_top_percentage}, var_bottom_n={var_bottom_n}:")
            
            # Filter each model's neurons
            filtered_neurons = {}
            for model_name, df in model_results.items():
                filtered_neurons[model_name] = filter_neurons(
                    df, 
                    norm_top_percentage=norm_top_percentage, 
                    var_bottom_n=var_bottom_n
                )
            
            # Compute IoU
            iou_result = compute_iou(filtered_neurons[model1], filtered_neurons[model2])
            print(f"IoU between {model1} and {model2}: {iou_result:.3f}")
            
            # For each filtered neuron, report its neuron_index and ratio
            print(f"Filtered neurons for {model1}: {list(filtered_neurons[model1].keys())}")
            print(f"Filtered neurons for {model2}: {list(filtered_neurons[model2].keys())}")
            
            # Store results
            all_results[analysis_key]["results"][param_key] = {
                "iou": float(iou_result),
                "neurons": {
                    model1: filtered_neurons[model1],
                    model2: filtered_neurons[model2]
                }
            }
    
    # Save all results to JSON
    save_results(all_results)
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze model neurons and calculate IoU.')
    parser.add_argument('--model1', type=str, required=True,
                        help='First model to analyze')
    parser.add_argument('--model2', type=str, required=True,
                        help='Second model to analyze')
    parser.add_argument('--norm_top_percentages', nargs='+', type=float, default=[0.25],
                        help='List of top percentage values for norm filtering')
    parser.add_argument('--var_bottom_ns', nargs='+', type=int, default=[10],
                        help='List of bottom N values for variance filtering')
    
    args = parser.parse_args()
    
    # Convert string inputs to appropriate types if needed
    norm_top_percentages = [float(p) for p in args.norm_top_percentages]
    var_bottom_ns = [int(n) for n in args.var_bottom_ns]
    
    # Run the main function
    main(args.model1, args.model2, norm_top_percentages, var_bottom_ns)


'''
python IoU.py --model1 llama-3.1-8b --model2 llama-3.1-8b-instruct --norm_top_percentages 0.25 --var_bottom_ns 10
'''

'''
python IoU.py --model1 llama-2-13b --model2 llama-2-13b-instruct --norm_top_percentages 0.25 --var_bottom_ns 10
'''