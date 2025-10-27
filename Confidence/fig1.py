
import os
import sys
import pandas as pd
import plotly.express as px
from utils import *
import plotly.graph_objects as go
import hydra
from omegaconf import DictConfig
import gc

def main():
    transformers_cache_dir = None
    #check if cuda is available
    if torch.cuda.is_available():
        device = 'cuda:0'

    else:
        device = 'cpu'
    os.chdir('../')


    #model_name = "meta-llama/llama-3.1-8b"
    model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    #model_name = "meta-llama/llama-3.1-8b-instruct"



    model, tokenizer = load_model_from_tl_name(model_name, device, transformers_cache_dir, hf_token=None)
    model = model.to(device)

    last_layer_neurons = model.W_out[-1]
    norm = torch.norm(last_layer_neurons, dim=1)
    unemb = model.W_U
    del model
    gc.collect()
    torch.cuda.empty_cache()
    comp_with_unemb = last_layer_neurons @ unemb
    normalized_composition = last_layer_neurons / last_layer_neurons.norm(dim=1).unsqueeze(1) @ unemb / unemb.norm(dim=0)
    comp_var = torch.var(comp_with_unemb, dim=1)
    cos_var = normalized_composition.var(dim=1)

    # make dataframe
    df = pd.DataFrame({'norm': norm.cpu().detach(),'cos_var': cos_var.cpu().detach()})

    # Add a new column 'neuron_index' to the DataFrame

    ylabel = 'LogitVar(w<sub>out</sub>)'
    xlabel = '||w<sub>out</sub>||'
    fig = px.scatter(df, x='norm', y='cos_var', color_discrete_sequence=['#636EFA'], labels={'norm':xlabel, 'cos_var': ylabel}, log_y=True, marginal_x='histogram', marginal_y='box')

    fig.update_layout(margin=dict(l=0, r=3, t=30, b=0))

    # decrease the width of the plot
    fig.update_layout(width=350, height=275)


    path = model_name.split("/")[-1]
    fig.write_image(f"./{path}_new.png")


if __name__ == "__main__":
    main()