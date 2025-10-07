import torch as t
import pandas as pd
import os
from glob import glob
import random
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25


def get_pcs(X, k=2, offset=0):
    """
    Performs Principal Component Analysis (PCA) on the n x d data matrix X. 
    Returns the k principal components, the corresponding eigenvalues and the projected data.
    """

    # Subtract the mean to center the data
    X = X - t.mean(X, dim=0)
    
    # Compute the covariance matrix
    cov_mat = t.mm(X.t(), X) / (X.size(0) - 1)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = t.linalg.eigh(cov_mat)
    
    # Since the eigenvalues and vectors are not necessarily sorted, we do that now
    sorted_indices = t.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the pcs
    eigenvectors = eigenvectors[:, offset:offset+k]
    
    return eigenvectors

def dict_recurse(d, f):
    """
    Recursively applies a function to a dictionary.
    """
    if isinstance(d, dict):
        out = {}
        for key in d:
            out[key] = dict_recurse(d[key], f)
        return out
    else:
        return f(d)

def collect_acts(dataset_name, model, layer, noperiod=False, center=True, scale=False, device='cpu'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(ROOT, 'acts', model)
    if noperiod:
        directory = os.path.join(directory, 'noperiod')
    directory = os.path.join(directory, dataset_name)
    activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    if len(activation_files) == 0:
        raise ValueError(f"Dataset {dataset_name} not found.")
    acts = [t.load(os.path.join(directory, f'layer_{layer}_{i}.pt')).to(device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)]
    acts = t.cat(acts, dim=0).float().to(device)
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts

def collect_polarity_data(dataset_names, model, layer, device='cpu'):
    """
    Takes as input the names of datasets in the format
    [affirmative_dataset1, negated_dataset1, affirmative_dataset2, negated_dataset2, ...]
    and returns a balanced training dataset of centered activations, activations, labels and polarities
    """

    all_acts_centered, all_acts, all_labels, all_polarities = [], [], [], []
    
    for dataset_name in dataset_names:
        directory = os.path.join(ROOT, 'acts', model)
        directory = os.path.join(directory, dataset_name)
        activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
        acts = [t.load(os.path.join(directory, f'layer_{layer}_{i}.pt')).to(device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)]
        acts = t.cat(acts, dim=0).float().to(device)
        all_acts.append(acts)
        labels = t.Tensor(pd.read_csv(f'datasets/{dataset_name}.csv')['label'].tolist()).to('cuda:0')
        all_labels.append(labels )
        
        polarity = -1.0 if 'neg_' in dataset_name else 1.0
        polarities = t.full((labels.shape[0],), polarity)
        all_polarities.append(polarities)

        all_acts_centered.append(acts - t.mean(acts, dim=0))

    return map(t.cat, (all_acts_centered, all_acts, all_labels, all_polarities))

def cat_data(d):
    """
    Given a dict of datasets (possible recursively nested), returns the concatenated activations and labels.
    """
    all_acts, all_labels = [], []
    for dataset in d:
        if isinstance(d[dataset], dict):
            if len(d[dataset]) != 0: # disregard empty dicts
                acts, labels = cat_data(d[dataset])
                all_acts.append(acts), all_labels.append(labels)
        else:
            acts, labels = d[dataset]
            all_acts.append(acts), all_labels.append(labels)
    return t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)

class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """
    def __init__(self):
        self.data = {
            'train' : {},
            'val' : {}
        } # dictionary of datasets
        self.proj = None # projection matrix for dimensionality reduction
    
    def add_dataset(self, dataset_name, model_size, layer, label='label', split=None, seed=None, noperiod=False, center=True, scale=False, device='cpu'):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        """
        acts = collect_acts(dataset_name, model_size, layer, noperiod=noperiod, center=center, scale=scale, device=device)
        df = pd.read_csv(os.path.join(ROOT, 'datasets', f'{dataset_name}.csv'))
        labels = t.Tensor(df[label].values).to(device)

        if split is None:
            self.data[dataset_name] = acts, labels

        if split is not None:
            assert 0 < split and split < 1
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(df)) < int(split * len(df))
            val = ~train
            self.data['train'][dataset_name] = acts[train], labels[train]
            self.data['val'][dataset_name] = acts[val], labels[val]

    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == 'all':
            data_dict = self.data
        elif datasets == 'train':
            data_dict = self.data['train']
        elif datasets == 'val':
            data_dict = self.data['val']
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data['train'][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data['val'][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets : self.data[datasets]}
        else:
            raise ValueError(f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}")
        acts, labels = cat_data(data_dict)
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        return acts, labels

    def set_pca(self, datasets, k=3, dim_offset=0):
        """
        Sets the projection matrix for dimensionality reduction by doing pca on the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        """
        acts, _ = self.get(datasets, proj=False)
        self.proj = get_pcs(acts, k=k, offset=dim_offset)

        self.data = dict_recurse(self.data, lambda x : (t.mm(x[0], self.proj), x[1]))
    



def collect_training_data(dataset_names, train_set_sizes, model_size, layer, **kwargs):
    """
    Takes as input the names of datasets in the format
    [affirmative_dataset1, negated_dataset1, affirmative_dataset2, negated_dataset2, ...]
    and returns a balanced training dataset of centered activations, activations, labels and polarities
    """
    all_acts_centered, all_acts, all_labels, all_polarities = [], [], [], []
    
    for dataset_name in dataset_names:
        dm = DataManager()
        dm.add_dataset(dataset_name,  model_size, layer, split=None, center=False, device='cpu')
        acts, labels = dm.data[dataset_name]
        
        polarity = -1.0 if 'neg_' in dataset_name else 1.0
        polarities = t.full((labels.shape[0],), polarity)

        # balance the training dataset by including an equal number of activations from each dataset
        # choose the same subset of statements for affirmative and negated version of the dataset
        if 'neg_' not in dataset_name:
            rand_subset = np.random.choice(acts.shape[0], min(train_set_sizes.values()), replace=False)
        
        all_acts_centered.append(acts[rand_subset, :] - t.mean(acts[rand_subset, :], dim=0))
        all_acts.append(acts[rand_subset, :])
        all_labels.append(labels[rand_subset])
        all_polarities.append(polarities[rand_subset])

    return map(t.cat, (all_acts_centered, all_acts, all_labels, all_polarities))
    
def compute_statistics(results):
    stats = {}
    for key in results:
        means = {dataset: np.mean(values) for dataset, values in results[key].items()}
        stds = {dataset: np.std(values) for dataset, values in results[key].items()}
        stats[key] = {'mean': means, 'std': stds}
    return stats

def compute_average_accuracies(results, num_iter):
    probe_stats = {}

    for probe_type in results:
        overall_means = []
        
        for i in range(num_iter):
            # Calculate mean accuracy for each dataset in this iteration
            iteration_means = [results[probe_type][dataset][i] for dataset in results[probe_type]]
            overall_means.append(np.mean(iteration_means))
        
        overall_means = np.array(overall_means)
        final_mean = np.mean(overall_means)
        std_dev = np.std(overall_means)
        
        probe_stats[probe_type.__name__] = {
            'mean': final_mean,
            'std_dev': std_dev
        }
    
    return probe_stats

def dataset_sizes(datasets):
    """
    Computes the size of each dataset, i.e. the number of statements.
    Input: array of strings that are the names of the datasets
    Output: dictionary, keys are the dataset names and values the number of statements
    """
    dataset_sizes_dict = {}
    for dataset in datasets:
        file_path = 'datasets/' + dataset + '.csv'
        with open(file_path, 'r') as file:
            line_count = sum(1 for line in file)
        dataset_sizes_dict[dataset] = line_count - 1
    return dataset_sizes_dict
