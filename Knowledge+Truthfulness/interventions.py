import torch as t
import pandas as pd
import os
import gc
from utils import collect_acts, collect_polarity_data
from generate_acts import load_model
from probes import LRProbe, MMProbe, CCSProbe, TTPD
import plotly.express as px
import json
import argparse
import configparser
import random

def intervention_experiment(model, queries, direction, hidden_states, intervention='none', batch_size=32, remote=True, get_completions=False, intervention_coefficient=1):
    """
    Add the direction to the specified hidden states and return the resulting probability diff P(TRUE) - P(FALSE)
    and sum P(TRUE) + P(FALSE) averaged over the data
    model : an nnsight LanguageModel
    queries : a list of statements to be labeled
    direction : a direction in the residual stream of the model
    hidden_states : list of (layer, -1 or 0) pairs, -1 for intervene before the period, 0 for intervene over the period
    get_completions : if True, return the model completions
    batch_size : batch size for forward passes
    remote : run on the NDIF server?
    intervention_coefficient : the coefficient for the intervention
    """
    assert intervention in ['none', 'add', 'subtract']

    true_idx, false_idx = model.tokenizer.encode(' TRUE')[-1], model.tokenizer.encode(' FALSE')[-1]
    len_suffix = len(model.tokenizer.encode('This statement is:'))

    p_diffs = []
    tots = []
    completions = []
    prob_max = []
    
    for batch_idx in range(0, len(queries), batch_size):
        batch = queries[batch_idx:batch_idx+batch_size]
        with model.trace(batch):
            for layer, offset in hidden_states:
                model.model.layers[layer].output[0][:,-len_suffix + offset, :] += \
                    direction * intervention_coefficient if intervention == 'add' else -direction * intervention_coefficient if intervention == 'subtract' else 0.
            logits = model.lm_head.output[:, -1, :]
            probs = logits.softmax(-1)
            diff = (probs[:, true_idx] - probs[:, false_idx]).save()
            tot = (probs[:, true_idx] + probs[:, false_idx]).save()
            most_likely_token_ids = t.argmax(probs, dim=1).save()
            prob = t.max(probs, dim=1).values.save()

                
        
                
        diff_d = diff.detach().cpu()
        del diff
        tot_d = tot.detach().cpu()
        del tot
        if get_completions:
            prob_d = prob.detach().cpu().tolist()
            prob_max.append(prob_d)
            most_likely_token_ids_d = most_likely_token_ids.detach().cpu().tolist()
            completion_tokens = [model.tokenizer.decode([token_id]) for token_id in most_likely_token_ids_d]
            completions.append(completion_tokens)
        p_diffs.append(diff_d)
        tots.append(tot_d)
        
        
    p_diffs = t.cat(p_diffs)
    tots = t.cat(tots)


    if get_completions:
        return p_diffs.mean().item(), tots.mean().item(), completions[0], prob_max[0]
    else:
        return p_diffs.mean().item(), tots.mean().item()

def prepare_data(prompts, dataset, subset='all', get_completions=False):
    """
    prompt : the few shot prompt
    dataset : dataset name
    subset : 'all', 'true', or 'false'
    sample_size : if specified, randomly sample this many prompts
    get_completions : if True, return the model completions
    Returns a list of queries to be run through the model for the patching experiment
    """
    seed = 1
    if get_completions:
        statements = dataset
    else:
        df = pd.read_csv(f'datasets/{dataset}.csv')
        if subset == 'all':
            statements = df['statement'].tolist()
        elif subset == 'true':
            statements = df[df['label'] == 1]['statement'].tolist()
        elif subset == 'false':
            statements = df[df['label'] == 0]['statement'].tolist()

    queries = []
    original_statements = []  
    
    for i, statement in enumerate(statements):

        random.seed(seed + i)
        

        shuffled_prompts = prompts.copy()
        random.shuffle(shuffled_prompts)
        
        concatenated_prompts = "\n".join(shuffled_prompts)
        
        query = concatenated_prompts + "\n" + statement + " This statement is:"
        
        if statement not in concatenated_prompts:
            queries.append(query)
            original_statements.append(statement)
            
    return queries, original_statements

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='llama-2-70b')
    parser.add_argument('--model_intervention', default='same')
    parser.add_argument('--probe', default='MMProbe')
    parser.add_argument('--train_datasets', nargs='+', default=['cities', 'neg_cities'], type=str)
    parser.add_argument('--val_dataset', default = 'sp_en_trans', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--intervention', default='none', type=str)
    parser.add_argument('--subset', default='all', type=str)
    parser.add_argument('--device', default='remote', type=str)
    parser.add_argument('--sample_size', default=3, type=int, help='Number of prompts to sample for examples')
    parser.add_argument('--get_completions', default=False, action='store_true')
    parser.add_argument('--intervention_coefficient', default=1, type=float)
    parser.add_argument('--results_file', default='experimental_outputs/label_change_intervention_results.json', type=str)
    args = parser.parse_args()
    get_completions = args.get_completions

    remote = args.device == 'remote'
    if args.model_intervention == 'same':
        model = load_model(args.model, args.device)
        inter_model = args.model
    else:
        model_1 = load_model(args.model, args.device)
        inter_model = args.model_intervention

    # prepare hidden states to intervene over
    config = configparser.ConfigParser()
    config.read('config.ini')
    start_layer = eval(config[inter_model]['intervene_layer'])
    end_layer = eval(config[inter_model]['probe_layer'])
    noperiod = eval(config[inter_model]['noperiod'])

    if noperiod:
        hidden_states = [
            (layer, -1) for layer in range(start_layer, end_layer + 1)
        ]
    else:
        hidden_states = []
        for layer in range(start_layer, end_layer + 1):
            hidden_states.append((layer, -1))
            hidden_states.append((layer, 0))
    
    print('training probe...')
    # get direction along which to intervene
    ProbeClass = eval(args.probe)
    if ProbeClass == LRProbe or ProbeClass == MMProbe or ProbeClass == 'random' or ProbeClass == TTPD:
        acts, labels = [], []
        for dataset in args.train_datasets:
            acts.append(collect_acts(dataset, args.model, end_layer, noperiod=noperiod).to('cuda:0'))
            labels.append(t.Tensor(pd.read_csv(f'datasets/{dataset}.csv')['label'].tolist()).to('cuda:0'))
        acts, labels = t.cat(acts), t.cat(labels)
        if ProbeClass == LRProbe or ProbeClass == MMProbe:
            probe = ProbeClass.from_data(acts, labels, device='cuda:0')
        elif ProbeClass == TTPD:
            acts_centered, acts, labels, polarities = collect_polarity_data(args.train_datasets, args.model, end_layer, device='cuda:0')
            probe = ProbeClass.from_data(acts_centered, acts, labels, polarities, device='cuda:0')
        elif ProbeClass == 'random':
            probe = MMProbe.from_data(acts, labels, device='cuda:0')
            probe.direction = t.nn.Parameter(t.randn_like(probe.direction))
    elif ProbeClass == CCSProbe:
        acts = collect_acts(args.train_datasets[0], args.model, end_layer, noperiod=noperiod).to('cuda:0')
        neg_acts = collect_acts(args.train_datasets[1], args.model, end_layer, noperiod=noperiod).to('cuda:0')
        labels = t.Tensor(pd.read_csv(f'datasets/{args.train_datasets[0]}.csv')['label'].tolist()).to('cuda:0')
        probe = ProbeClass.from_data(acts, neg_acts, labels=labels, device='cuda:0')

    if ProbeClass == TTPD:
        direction = t.tensor(probe.t_g).cuda()
    else:
        direction = probe.direction
    true_acts, false_acts = acts[labels==1], acts[labels==0]
    true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
    direction = direction / direction.norm()
    diff = (true_mean - false_mean) @ direction
    direction = diff * direction
    direction_m1 = direction.cpu()


    ProbeClass = eval(args.probe)
    if ProbeClass == LRProbe or ProbeClass == MMProbe or ProbeClass == 'random' or ProbeClass == TTPD:
        acts, labels = [], []
        for dataset in args.train_datasets:
            acts.append(collect_acts(dataset, args.model_intervention, end_layer, noperiod=noperiod).to('cuda:0'))
            labels.append(t.Tensor(pd.read_csv(f'datasets/{dataset}.csv')['label'].tolist()).to('cuda:0'))
        acts, labels = t.cat(acts), t.cat(labels)
        if ProbeClass == LRProbe or ProbeClass == MMProbe:
            probe = ProbeClass.from_data(acts, labels, device='cuda:0')
        elif ProbeClass == TTPD:
            acts_centered, acts, labels, polarities = collect_polarity_data(args.train_datasets, args.model_intervention, end_layer, device='cuda:0')
            probe = ProbeClass.from_data(acts_centered, acts, labels, polarities, device='cuda:0')
        elif ProbeClass == 'random':
            probe = MMProbe.from_data(acts, labels, device='cuda:0')
            probe.direction = t.nn.Parameter(t.randn_like(probe.direction))
    elif ProbeClass == CCSProbe:
        acts = collect_acts(args.train_datasets[0], args.model_intervention, end_layer, noperiod=noperiod).to('cuda:0')
        neg_acts = collect_acts(args.train_datasets[1], args.model_intervention, end_layer, noperiod=noperiod).to('cuda:0')
        labels = t.Tensor(pd.read_csv(f'datasets/{args.train_datasets[0]}.csv')['label'].tolist()).to('cuda:0')
        probe = ProbeClass.from_data(acts, neg_acts, labels=labels, device='cuda:0')

    if ProbeClass == TTPD:
        direction = t.tensor(probe.t_g).cuda()
    else:
        direction = probe.direction
    true_acts, false_acts = acts[labels==1], acts[labels==0]
    true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
    direction = direction / direction.norm()
    diff = (true_mean - false_mean) @ direction
    direction = diff * direction
    direction_m2 = direction.cpu()

    prompts = []
    with open('datasets/few_shot_samples.json', 'r') as f:
        data = json.load(f)
        for dataset in data:
            if dataset["dataset_name"] == args.val_dataset:
                prompts = dataset["few_shot_examples"]
                continue
                
    # prepare data
    queries_true, statements_true = prepare_data(prompts, args.val_dataset, subset='true')
    queries_false, statements_false = prepare_data(prompts, args.val_dataset, subset='false')


    true_statements = [
        "The city of Paris is in France.",
        "The city of Beijing is in China."
    ]
    
    false_statements = [
        "The city of Paris is in China.",
        "The city of Beijing is in France."
    ]

    sample_queries_true, sample_statements_true = prepare_data(prompts, true_statements, subset='true', get_completions=True)
    sample_queries_false, sample_statements_false = prepare_data(prompts, false_statements, subset='false', get_completions=True)
    
    

    print('running intervention experiment...')
    # do intervention experiment
    p_diff_m1_true_base, _, = intervention_experiment(model_1, queries_true, direction_m1, hidden_states,
                                          intervention='none', batch_size=args.batch_size, remote=remote)
    p_diff_m1_false_base, _, = intervention_experiment(model_1, queries_false, direction_m1, hidden_states,
                                          intervention='none', batch_size=args.batch_size, remote=remote)
    
    # Get sample completions
    _, _, m1_true_base_completions, m1_true_base_prob = intervention_experiment(model_1, sample_queries_true, direction_m1, hidden_states,
                                          intervention='none', batch_size=args.batch_size, remote=remote, get_completions=True)
    _, _, m1_false_base_completions, m1_false_base_prob = intervention_experiment(model_1, sample_queries_false, direction_m1, hidden_states,
                                          intervention='none', batch_size=args.batch_size, remote=remote, get_completions=True)

    p_diff_m1_m1_true_inter, _, = intervention_experiment(model_1, queries_true, direction_m1, hidden_states,
                                          intervention='subtract', batch_size=args.batch_size, remote=remote, intervention_coefficient=args.intervention_coefficient)
    p_diff_m1_m1_false_inter, _, = intervention_experiment(model_1, queries_false, direction_m1, hidden_states,
                                          intervention='add', batch_size=args.batch_size, remote=remote, intervention_coefficient=args.intervention_coefficient)
    
    # Get sample completions for m1_m1 interventions
    _, _, m1_m1_true_inter_completions, m1_m1_true_prob = intervention_experiment(model_1, sample_queries_true, direction_m1, hidden_states,
                                          intervention='subtract', batch_size=args.batch_size, remote=remote, get_completions=True, intervention_coefficient=args.intervention_coefficient)
    _, _, m1_m1_false_inter_completions, m1_m1_false_prob = intervention_experiment(model_1, sample_queries_false, direction_m1, hidden_states,
                                          intervention='add', batch_size=args.batch_size, remote=remote, get_completions=True, intervention_coefficient=args.intervention_coefficient)
    
    p_diff_m2_m1_true_inter, _, = intervention_experiment(model_1, queries_true, direction_m2, hidden_states,
                                          intervention='subtract', batch_size=args.batch_size, remote=remote, intervention_coefficient=args.intervention_coefficient)
    p_diff_m2_m1_false_inter, _, = intervention_experiment(model_1, queries_false, direction_m2, hidden_states,
                                          intervention='add', batch_size=args.batch_size, remote=remote, intervention_coefficient=args.intervention_coefficient)
    
    # Get sample completions for m2_m1 interventions
    _, _, m2_m1_true_inter_completions, m2_m1_true_prob = intervention_experiment(model_1, sample_queries_true, direction_m2, hidden_states,
                                          intervention='subtract', batch_size=args.batch_size, remote=remote, get_completions=True, intervention_coefficient=args.intervention_coefficient)
    _, _, m2_m1_false_inter_completions, m2_m1_false_prob = intervention_experiment(model_1, sample_queries_false, direction_m2, hidden_states,
                                          intervention='add', batch_size=args.batch_size, remote=remote, get_completions=True, intervention_coefficient=args.intervention_coefficient)
    
    del model_1
    gc.collect()
    t.cuda.empty_cache()
    
    model_2 = load_model(args.model_intervention, args.device)
    
    p_diff_m2_true_base, _, = intervention_experiment(model_2, queries_true, direction_m1, hidden_states,
                                          intervention='none', batch_size=args.batch_size, remote=remote)
    p_diff_m2_false_base, _, = intervention_experiment(model_2, queries_false, direction_m1, hidden_states,
                                          intervention='none', batch_size=args.batch_size, remote=remote)
    
    # Get sample completions for model 2 base
    _, _, m2_true_base_completions, m2_true_base_probe = intervention_experiment(model_2, sample_queries_true, direction_m1, hidden_states,
                                          intervention='none', batch_size=args.batch_size, remote=remote, get_completions=True)
    _, _, m2_false_base_completions, m2_false_base_probe = intervention_experiment(model_2, sample_queries_false, direction_m1, hidden_states,
                                          intervention='none', batch_size=args.batch_size, remote=remote, get_completions=True)
    
    p_diff_m2_m2_true_inter, _, = intervention_experiment(model_2, queries_true, direction_m2, hidden_states,
                                          intervention='subtract', batch_size=args.batch_size, remote=remote, intervention_coefficient=args.intervention_coefficient)
    p_diff_m2_m2_false_inter, _, = intervention_experiment(model_2, queries_false, direction_m2, hidden_states,
                                          intervention='add', batch_size=args.batch_size, remote=remote, intervention_coefficient=args.intervention_coefficient)
    
    # Get sample completions for m2_m2 interventions
    _, _, m2_m2_true_inter_completions, m2_m2_true_prob = intervention_experiment(model_2, sample_queries_true, direction_m2, hidden_states,
                                          intervention='subtract', batch_size=args.batch_size, remote=remote, get_completions=True, intervention_coefficient=args.intervention_coefficient)
    _, _, m2_m2_false_inter_completions, m2_m2_false_prob = intervention_experiment(model_2, sample_queries_false, direction_m2, hidden_states,
                                          intervention='add', batch_size=args.batch_size, remote=remote, get_completions=True, intervention_coefficient=args.intervention_coefficient)
    
    p_diff_m1_m2_true_inter, _, = intervention_experiment(model_2, queries_true, direction_m1, hidden_states,
                                          intervention='subtract', batch_size=args.batch_size, remote=remote, intervention_coefficient=args.intervention_coefficient)
    p_diff_m1_m2_false_inter, _, = intervention_experiment(model_2, queries_false, direction_m1, hidden_states,
                                          intervention='add', batch_size=args.batch_size, remote=remote, intervention_coefficient=args.intervention_coefficient)
    
    # Get sample completions for m1_m2 interventions
    _, _, m1_m2_true_inter_completions, m1_m2_true_prob = intervention_experiment(model_2, sample_queries_true, direction_m1, hidden_states,
                                          intervention='subtract', batch_size=args.batch_size, remote=remote, get_completions=True, intervention_coefficient=args.intervention_coefficient)
    _, _, m1_m2_false_inter_completions, m1_m2_false_prob = intervention_experiment(model_2, sample_queries_false, direction_m1, hidden_states,
                                          intervention='add', batch_size=args.batch_size, remote=remote, get_completions=True, intervention_coefficient=args.intervention_coefficient)
    
    nif = False
    
    if nif:

        nif_m1_m1_true_false = (p_diff_m1_m1_true_inter - p_diff_m1_true_base) / (p_diff_m1_false_base - p_diff_m1_true_base)
        nif_m1_m1_false_true = (p_diff_m1_m1_false_inter - p_diff_m1_false_base) / (p_diff_m1_true_base - p_diff_m1_false_base)

        nif_m2_m1_true_false = (p_diff_m2_m1_true_inter - p_diff_m1_true_base) / (p_diff_m1_false_base - p_diff_m1_true_base)
        nif_m2_m1_false_true = (p_diff_m2_m1_false_inter - p_diff_m1_false_base) / (p_diff_m1_true_base - p_diff_m1_false_base)

        nif_m1_m2_true_false = (p_diff_m1_m2_true_inter - p_diff_m2_true_base) / (p_diff_m2_false_base - p_diff_m2_true_base)
        nif_m1_m2_false_true = (p_diff_m1_m2_false_inter - p_diff_m2_false_base) / (p_diff_m2_true_base - p_diff_m2_false_base)

        nif_m2_m2_true_false = (p_diff_m2_m2_true_inter - p_diff_m2_true_base) / (p_diff_m2_false_base - p_diff_m2_true_base)
        nif_m2_m2_false_true = (p_diff_m2_m2_false_inter - p_diff_m2_false_base) / (p_diff_m2_true_base - p_diff_m2_false_base)
    else:
        nif_m1_m1_true_false = (p_diff_m1_m1_true_inter - p_diff_m1_true_base) / (-1 - p_diff_m1_true_base)
        nif_m1_m1_false_true = (p_diff_m1_m1_false_inter - p_diff_m1_false_base) / (1- p_diff_m1_false_base)

        nif_m2_m1_true_false = (p_diff_m2_m1_true_inter - p_diff_m1_true_base) / (-1 - p_diff_m1_true_base)
        nif_m2_m1_false_true = (p_diff_m2_m1_false_inter - p_diff_m1_false_base) / (1 - p_diff_m1_false_base)

        nif_m1_m2_true_false = (p_diff_m1_m2_true_inter - p_diff_m2_true_base) / (-1 - p_diff_m2_true_base)
        nif_m1_m2_false_true = (p_diff_m1_m2_false_inter - p_diff_m2_false_base) / (1 - p_diff_m2_false_base)

        nif_m2_m2_true_false = (p_diff_m2_m2_true_inter - p_diff_m2_true_base) / (-1 - p_diff_m2_true_base)
        nif_m2_m2_false_true = (p_diff_m2_m2_false_inter - p_diff_m2_false_base) / (1 - p_diff_m2_false_base)

    # Convert list of tuples to list of dictionaries for better JSON readability
    def format_statement_completions(statements, completions, probs):
        return [{"statement": stmt, "completion": comp, "prob":prob} for stmt, comp, prob in zip(statements, completions, probs)]
    
    # Create sample outputs
    sample_outputs = {
        "model_1_base": {
            "true_statements": format_statement_completions(sample_statements_true, m1_true_base_completions, m1_true_base_prob),
            "false_statements": format_statement_completions(sample_statements_false, m1_false_base_completions, m1_false_base_prob)
        },
        "model_2_base": {
            "true_statements": format_statement_completions(sample_statements_true, m2_true_base_completions, m2_true_base_probe),
            "false_statements": format_statement_completions(sample_statements_false, m2_false_base_completions, m2_false_base_probe)
        },
        "m1_m1_intervention": {
            "true_statements": format_statement_completions(sample_statements_true, m1_m1_true_inter_completions, m1_m1_true_prob),
            "false_statements": format_statement_completions(sample_statements_false, m1_m1_false_inter_completions, m1_m1_false_prob)
        },
        "m2_m1_intervention": {
            "true_statements": format_statement_completions(sample_statements_true, m2_m1_true_inter_completions, m2_m1_true_prob),
            "false_statements": format_statement_completions(sample_statements_false, m2_m1_false_inter_completions, m2_m1_false_prob)
        },
        "m1_m2_intervention": {
            "true_statements": format_statement_completions(sample_statements_true, m1_m2_true_inter_completions, m1_m2_true_prob),
            "false_statements": format_statement_completions(sample_statements_false, m1_m2_false_inter_completions, m1_m2_false_prob)
        },
        "m2_m2_intervention": {
            "true_statements": format_statement_completions(sample_statements_true, m2_m2_true_inter_completions, m2_m2_true_prob),
            "false_statements": format_statement_completions(sample_statements_false, m2_m2_false_inter_completions, m2_m2_false_prob)
        }
    }

    # save results

    if get_completions:

        out = {
            'model_1' : args.model,
            'model_2' : inter_model,
            'train_datasets' : args.train_datasets,
            'val_dataset' : args.val_dataset,
            'probe class' : ProbeClass.__name__,
            'prompt' : prompts,
            'nif_m1_m1_true_false' : nif_m1_m1_true_false,
            'nif_m1_m1_false_true' : nif_m1_m1_false_true,

            'nif_m2_m1_true_false' : nif_m2_m1_true_false,
            'nif_m2_m1_false_true' : nif_m2_m1_false_true,

            'nif_m1_m2_true_false' : nif_m1_m2_true_false,
            'nif_m1_m2_false_true' : nif_m1_m2_false_true,

            'nif_m2_m2_true_false' : nif_m2_m2_true_false,
            'nif_m2_m2_false_true' : nif_m2_m2_false_true,

            'p_diff_m1_true_base' : p_diff_m1_true_base,
            'p_diff_m1_false_base' : p_diff_m1_false_base,
            'p_diff_m2_true_base' : p_diff_m2_true_base,
            'p_diff_m2_false_base' : p_diff_m2_false_base,
            
            'sample_outputs': sample_outputs,
            'intervention_coefficient' : args.intervention_coefficient
        }
    else:
        out = {
            'model_1' : args.model,
            'model_2' : inter_model,
            'train_datasets' : args.train_datasets,
            'val_dataset' : args.val_dataset,
            'probe class' : ProbeClass.__name__,
            'prompt' : prompts,
            'nif_m1_m1_true_false' : nif_m1_m1_true_false,
            'nif_m1_m1_false_true' : nif_m1_m1_false_true,
            'avg_nif_m1_m1' : (nif_m1_m1_true_false + nif_m1_m1_false_true) / 2,

            'nif_m2_m1_true_false' : nif_m2_m1_true_false,
            'nif_m2_m1_false_true' : nif_m2_m1_false_true,
            'avg_nif_m2_m1' : (nif_m2_m1_true_false + nif_m2_m1_false_true) / 2,

            'nif_m1_m2_true_false' : nif_m1_m2_true_false,
            'nif_m1_m2_false_true' : nif_m1_m2_false_true,
            'avg_nif_m1_m2' : (nif_m1_m2_true_false + nif_m1_m2_false_true) / 2,

            'nif_m2_m2_true_false' : nif_m2_m2_true_false,
            'nif_m2_m2_false_true' : nif_m2_m2_false_true,
            'avg_nif_m2_m2' : (nif_m2_m2_true_false + nif_m2_m2_false_true) / 2,

            'p_diff_m1_true_base' : p_diff_m1_true_base,
            'p_diff_m1_false_base' : p_diff_m1_false_base,
            'p_diff_m2_true_base' : p_diff_m2_true_base,
            'p_diff_m2_false_base' : p_diff_m2_false_base,
            'intervention_coefficient' : args.intervention_coefficient
        }

    with open(args.results_file, 'r') as f:
        data = json.load(f)
    data.append(out)
    with open(args.results_file, 'w') as f:
        json.dump(data, f, indent=4)
