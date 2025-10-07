# Refusal Direction Extraction and Application

This directory contains code for extracting and applying refusal directions to language models. The refusal direction is a vector that represents the difference between how a model responds to harmful vs harmless instructions, which can be used to steer model behavior.

## Overview

The workflow consists of two main steps:
1. **Extract refusal directions** from models using contrastive pairs of harmful and harmless instructions
2. **Apply refusal directions** to steer model behavior through activation interventions

## Files

- `extract_refusal_dir.py` - Extracts refusal directions from various models
- `apply_refusal_dir.py` - Applies refusal directions to steer model behavior
- `utils.py` - Helper functions for data loading, generation, and scoring
- `refusal_directions/` - Directory containing extracted refusal direction vectors
- `results/` - Directory containing experimental results

## Usage

### Step 1: Extract Refusal Directions

First, extract refusal directions from the models:

```bash
# Extract from LLaMA 3.1 8B base model
python extract_refusal_dir.py --model_name llama-3.1-8b --n_inst_train 128

# Extract from LLaMA 3.1 8B Instruct model
python extract_refusal_dir.py --model_name llama-3.1-8b-instruct --n_inst_train 128

# Extract from Tulu-sft model
python extract_refusal_dir.py --model_name tulu --n_inst_train 128
```

### Step 2: Apply Refusal Directions

Apply the extracted refusal directions to steer model behavior:

```bash
# Apply both base model and instruct model refusal direciton to do intervention on base model
python apply_refusal_dir.py \
    --model_name llama-3.1-8b \
    --instruction_type harmful \
    --intervention_type ablation 

# Apply both base model and intruct model refusal direciton to do intervention on instruct model
python apply_refusal_dir.py \
    --model_name llama-3.1-8b \
    --instruction_type harmless \
    --intervention_type addition 
    --intervene_chat \

# Apply both base model and sft model refusal direciton to do intervention on sft model
python apply_refusal_dir.py \
    --model_name llama-3.1-8b \
    --instruction_type harmless \
    --intervention_type addition \
    --use_sft \
    --intervene_chat 

# Apply both instruct model and sft model refusal direciton to do intervention on sft model
python apply_refusal_dir.py \
    --model_name llama-3.1-8b \
    --instruction_type harmless \
    --intervention_type addition \
    --sft_int 

# Apply both instruct model and sft model refusal direciton to do intervention on instruct model
python apply_refusal_dir.py \
    --model_name llama-3.1-8b \
    --instruction_type harmless \
    --intervention_type addition \
    --sft_int \
    --intervene_chat 
```

**Key Parameters:**
- `--model_name`: Base model name (e.g., `llama-3.1-8b`)
- `--instruction_type`: Type of instructions (`harmful` or `harmless`)
- `--intervention_type`: Type of intervention (`addition`, `subtraction`, or `ablation`)
- `--intervene_chat`: Whether to intervene on chat model (if false, intervenes on base model)
- `--use_sft`: Whether to use SFT checkpoint (Tulu) as the chat model
- `==sft_int`: Whether to do experiments between sft and instruct model


This directory is adapted from https://github.com/ckkissane/base-models-refuse?tab=readme-ov-file
