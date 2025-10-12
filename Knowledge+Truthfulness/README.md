# Truthfulness Probe Transfer and Intervention Trandfer Experiments + Knowledge Patching

This directory contains code for training truthfulness probes and conducting intervention experiments on language models, with a focus on LLaMA-3.1-8B and LLaMA-3.1-8B-Instruct models.


### Model Configuration

The experiments are configured in `config.ini`. For LLaMA-3.1-8B models:

```ini
[llama-3.1-8b]
weights_directory = meta-llama/llama-3.1-8b
name = LLaMA-3.1-8B
probe_layer = 12
intervene_layer = 8
noperiod = False

[llama-3.1-8b-instruct]
weights_directory = meta-llama/llama-3.1-8b-instruct
name = LLaMA-3.1-8B-instruct
probe_layer = 12
intervene_layer = 8
noperiod = False
```


## Step 1: Generate Activations

Before training probes, you need to extract activations from the target layers.

### For LLaMA-3.1-8B (Base Model)

```bash
python generate_acts.py --model llama-3.1-8b --layers 8 9 10 11 12 \
--datasets cities sp_en_trans animal_class inventors element_symb facts --device cuda:0
```

### For LLaMA-3.1-8B-Instruct

```bash
python generate_acts.py --model llama-3.1-8b-instruct --layers 8 9 10 11 12 \
--datasets cities sp_en_trans animal_class inventors element_symb facts --device cuda:0
```


### Parameters Explained

- `--model`: Model name from config.ini
- `--layers`: Target layers for activation extraction (8-13 for LLaMA-3.1-8B)
- `--datasets`: Datasets to process (see `datasets/` directory)

## Step 2: Probe Transfer 


```bash
# probe transfer evaluated on cities dataset
python -W ignore probe_transfer.py --model1 llama-3.1-8b --model2 llama-3.1-8b-instruct \
--train_datasets_m1 sp_en_trans animal_class inventors element_symb facts  --train_datasets_m2 sp_en_trans animal_class inventors element_symb facts \
--val_datasets_m1 cities --val_datasets_m2 cities 
```


## Step 3: Cross Intervention Experiments


```bash
# full model intervention experiment on citites dataset
python interventions.py --model llama-3.1-8b --model_intervention llama-3.1-8b-instruct --probe MMProbe --device cuda:0 \
--train_datasets sp_en_trans animal_class inventors element_symb facts --val_dataset cities
```

---

# Patching Experiments

This section covers how to run patching experiments to understand where knowledge is encoded in model representations.

## Data Preparation Steps

Before running patching experiments, you need to process the raw CSV datasets to create paired true/false statements that are matched by their last word. This is crucial for the patching experiments.

**For each dataset and model combination, run:**

```bash
# For LLaMA-3.1-8B
python patching_process_data.py --model llama-3.1-8b --dataset cities --device cuda:0
python patching_process_data.py --model llama-3.1-8b --dataset neg_cities --device cuda:0
python patching_process_data.py --model llama-3.1-8b --dataset sp_en_trans --device cuda:0
python patching_process_data.py --model llama-3.1-8b --dataset neg_sp_en_trans --device cuda:0
python patching_process_data.py --model llama-3.1-8b --dataset larger_than --device cuda:0
python patching_process_data.py --model llama-3.1-8b --dataset smaller_than --device cuda:0
```

This will create `*_paired.json` files in the datasets directory with matched true/false statement pairs.


## Running Patching Experiments

### Single Model Patching

Patching within one model (replacing activations from true statements with those from false statements):

```bash
python patching.py --model llama-3.1-8b-instruct --dataset cities --classes true_false --device cuda:0
```

### Cross-Model Patching

Patching between two models (replacing activations of true statements from one model with activations of false statements from another model):

```bash
python patching.py --model llama-3.1-8b llama-3.1-8b-instruct --dataset cities --classes true_false --device cuda:0
```

## Visualizing Results

After running patching experiments, you can visualize the results.

### Visualizing One Patching Result

```bash
python patching_visualize.py --dataset cities --classes true_false --model_names1 llama-3.1-8b
```

If you want to visualize the cross-model patching result, please write the two model's names after the "--model_names1", such as "--model_names1 llama-3.1-8b llama-3.1-8b-instruct" (not "--model_names2").

```bash
python patching_visualize.py --dataset cities --classes true_false --model_names1 llama-3.1-8b
```

### Visualizing the Comparison of Two Patching Results

```bash
python patching_visualize.py --dataset cities --classes true_false --model_names1 llama-3.1-8b --model_names2 llama-3.1-8b-instruct
```


This directory is adapted from https://github.com/saprmarks/geometry-of-truth
