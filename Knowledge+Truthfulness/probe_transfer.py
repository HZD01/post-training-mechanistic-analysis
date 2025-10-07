import torch as t
from utils import DataManager
import random
import matplotlib.pyplot as plt
import random
from probes import MMProbe
import configparser
import json
import argparse
import os

config = configparser.ConfigParser()
config.read('config.ini')
split = 0.8

def train_and_evaluate_probes(model1, model2, datasets_m1, datasets_m2, val_datasets_m1, val_datasets_m2, layer=-1, use_full_train=True):
    device = 'cuda:0' if t.cuda.is_available() else 'cpu'
    ProbeClass = MMProbe
    if layer == -1:
        layer = eval(config[model1]['probe_layer'])
    noperiod = eval(config[model1]['noperiod'])
    
    # Set split based on use_full_train parameter
    train_split = None if use_full_train else split
    
    # Train probe1 on model1 with datasets_m1
    dm1 = DataManager()
    for dataset in datasets_m1:
        dm1.add_dataset(dataset, model1, layer, split=train_split, noperiod=noperiod, device=device)
    
    if use_full_train:
        train_acts, train_labels = dm1.get('all')
    else:
        train_acts, train_labels = dm1.get('train')
    
    probe1 = ProbeClass.from_data(train_acts, train_labels, device=device)
    direction1 = probe1.direction
    

    # Train probe2 on model2 with datasets_m2
    dm2 = DataManager()
    for dataset in datasets_m2:
        dm2.add_dataset(dataset, model2, layer, split=train_split, noperiod=noperiod, device=device)
    
    if use_full_train:
        train_acts, train_labels = dm2.get('all')
    else:
        train_acts, train_labels = dm2.get('train')
    
    probe2 = ProbeClass.from_data(train_acts, train_labels, device=device)
    direction2 = probe2.direction

    # Calculate and print cosine similarity between the two directions
    cosine_sim = t.nn.functional.cosine_similarity(direction1, direction2, dim=0).item()
    print(f"Cosine similarity between probe directions: {cosine_sim}")

    # Only evaluate on validation set if not using full train
    if not use_full_train:
        val_acts1, val_labels1 = dm1.get('val')
        val_preds = (probe1.pred(val_acts1, iid=True) == val_labels1).float().mean().item()
        print(f"Validation accuracy for training of probe1: {val_preds}")

        val_acts2, val_labels2 = dm2.get('val')
        val_preds = (probe2.pred(val_acts2, iid=True) == val_labels2).float().mean().item()
        print(f"Validation accuracy for training of probe2: {val_preds}")

        val_preds = (probe1.pred(val_acts2, iid=True) == val_labels2).float().mean().item()
        print(f"Validation accuracy for training for probe1 on model2: {val_preds}")

        val_preds = (probe2.pred(val_acts1, iid=True) == val_labels1).float().mean().item()
        print(f"Validation accuracy for training for probe2 on model1: {val_preds}")

    # Evaluate on validation datasets for model1
    dm_val1 = DataManager()
    for dataset in val_datasets_m1:
        dm_val1.add_dataset(dataset, model1, layer, split=None, noperiod=noperiod, device=device)
        acts1, labels1 = dm_val1.data[dataset]
        preds1 = probe1.pred(acts1, iid=True)
        preds2 = probe2.pred(acts1, iid=True)
        acc1 = (preds1 == labels1).float().mean().item()
        acc2 = (preds2 == labels1).float().mean().item()

        print(f"Accuracy for probe1 on model1 on {dataset}: {acc1}")
        print(f"Accuracy for probe2 on model1 on {dataset}: {acc2}")

    # Evaluate on validation datasets for model2
    dm_val2 = DataManager()
    for dataset in val_datasets_m2:
        dm_val2.add_dataset(dataset, model2, layer, split=None, noperiod=noperiod, device=device)
        acts2, labels2 = dm_val2.data[dataset]
        preds1 = probe1.pred(acts2, iid=True)
        preds2 = probe2.pred(acts2, iid=True)
        acc1 = (preds1 == labels2).float().mean().item()
        acc2 = (preds2 == labels2).float().mean().item()

        print(f"Accuracy for probe1 on model2 on {dataset}: {acc1}")
        print(f"Accuracy for probe2 on model2 on {dataset}: {acc2}")
    
    return probe1, probe2, direction1, direction2
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default='llama-3.1-8b')
    parser.add_argument('--model2', type=str, default='llama-3.1-8b-instruct')
    parser.add_argument('--train_datasets_m1', nargs='+', type=str, default=["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans"])
    parser.add_argument('--train_datasets_m2', nargs='+', type=str, default=["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans"])
    parser.add_argument('--val_datasets_m1', nargs='+', type=str, default=["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans"])
    parser.add_argument('--val_datasets_m2', nargs='+', type=str, default=["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans"])
    parser.add_argument('--layer', type=int, default=-1)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    train_and_evaluate_probes(args.model1, args.model2, args.train_datasets_m1, args.train_datasets_m2, args.val_datasets_m1, args.val_datasets_m2, args.layer)

## sample command:
'''
python -W ignore plot_probes_results.py --model1 llama-3.1-8b --model2 llama-3.1-8b-instruct \
--train_datasets_m1 refusal_train_base  --train_datasets_m2 refusal_train_instruct \
--val_datasets_m1 refusal_test_base --val_datasets_m2 refusal_test_instruct

python -W ignore plot_probes_results.py --model1 llama-3.1-8b --model2 llama-3.1-tulu3-8b-sft \
--train_datasets_m1 refusal_train_base  --train_datasets_m2 refusal_train_tulu \
--val_datasets_m1 refusal_test_base --val_datasets_m2 refusal_test_tulu
'''


'''


python -W ignore plot_probes_results.py --model1 llama-3.1-8b --model2 llama-3.1-8b-instruct \
--train_datasets_m1 sp_en_trans animal_class inventors element_symb facts neg_facts   --train_datasets_m2 sp_en_trans neg_sp_en_trans animal_class neg_animal_class inventors neg_inventors element_symb neg_element_symb facts neg_facts \
--val_datasets_m1 cities neg_cities --val_datasets_m2 cities neg_cities

python -W ignore plot_probes_results.py --model1 llama-2-13b --model2 llama-2-13b-instruct \
--datasets_m1 sp_en_trans animal_class inventors element_symb facts   --datasets_m2 sp_en_trans animal_class inventors element_symb facts \
--val_datasets_m1 cities --val_datasets_m2 cities

python -W ignore plot_probes_results.py --model1 llama-2-13b --model2 llama-2-13b-instruct \
--datasets_m1 cities animal_class inventors element_symb facts   --datasets_m2 cities animal_class inventors element_symb facts \
--val_datasets_m1 sp_en_trans --val_datasets_m2 sp_en_trans

python -W ignore plot_probes_results.py --model1 llama-2-13b --model2 llama-2-13b-instruct \
--datasets_m1 cities sp_en_trans inventors element_symb facts   --datasets_m2 cities sp_en_trans inventors element_symb facts \
--val_datasets_m1 animal_class --val_datasets_m2 animal_class

python -W ignore plot_probes_results.py --model1 llama-2-13b --model2 llama-2-13b-instruct \
--datasets_m1 cities sp_en_trans animal_class element_symb facts  --datasets_m2 cities sp_en_trans animal_class element_symb facts \
--val_datasets_m1 inventors --val_datasets_m2 inventors

python -W ignore plot_probes_results.py --model1 llama-2-13b --model2 llama-2-13b-instruct \
--datasets_m1 cities sp_en_trans animal_class inventors facts   --datasets_m2 cities sp_en_trans animal_class inventors facts    \
--val_datasets_m1 element_symb --val_datasets_m2 element_symb

python -W ignore plot_probes_results.py --model1 llama-2-13b --model2 llama-2-13b-instruct \
--datasets_m1 cities sp_en_trans animal_class inventors element_symb   --datasets_m2 cities sp_en_trans animal_class inventors element_symb    \
--val_datasets_m1 facts  --val_datasets_m2 facts 

python -W ignore plot_probes_results.py --model1 llama-2-13b --model2 llama-2-13b-instruct \
--datasets_m1 cities sp_en_trans animal_class inventors element_symb facts   --datasets_m2 cities sp_en_trans animal_class inventors element_symb facts    \
--val_datasets_m1 facts  --val_datasets_m2 facts 
'''