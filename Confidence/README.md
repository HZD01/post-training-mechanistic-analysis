# Entropy Neurons Exoeriments


## Setup

Install required packages
```
pip install git+https://github.com/neelnanda-io/neel-plotly.git
pip install git+https://github.com/neelnanda-io/neelutils.git
```

##

To plot distribution of logitvar vs weigth norm, run `python fig1.py`.

To select and compare entropy neuron candidates between two model checkpoints run
```
python IoU.py --model1 llama-3.1-8b --model2 llama-3.1-8b-instruct --norm_top_percentages 0.25 --var_bottom_ns 10
```

- `--model1`, `--model2` specify the names of the models to compare (should correspond to names present in the neuron analysis results files).
- `--norm_top_percentages` sets the top percentage(s) of neurons (by output weight norm) to select as candidates (e.g., 0.25 for top 25%).
- `--var_bottom_ns` sets how many neurons to select from the bottom based on logit variance (e.g., 10 means pick the 10 lowest-variance neurons among those passing the norm filter).


The code of this directory is adapted from https://github.com/bpwu1/confidence-regulation-neurons

