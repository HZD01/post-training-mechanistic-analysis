import torch
import functools
from colorama import Fore
import argparse
import textwrap
from jaxtyping import Float
import einops
from torch import Tensor
import matplotlib.pyplot as plt
import os
import numpy as np

from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens.hook_points import HookPoint

from utils import get_harmful_instructions, get_harmless_instructions, tokenize_instructions, get_generations, get_refusal_scores, save_records_to_json, GEMMA_CHAT_TEMPLATE, GEMMA_BASE_TEMPLATE, QWEN_CHAT_TEMPLATE, LLAMA_CHAT_TEMPLATE, LLAMA_BASE_TEMPLATE, LLAMA3_CHAT_TEMPLATE, LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM, LLAMA3_BASE_TEMPLATE, TULU_TEMPLATE, MISTRAL_TEMPLATE, LLAMA2_CHAT_TEMPLATE, LLAMA2_BASE_TEMPLATE

def act_add_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    steering_coef: int
):
    activation += steering_coef * direction
    return activation

def subtract_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    steering_coef: int
):
    activation -= steering_coef * direction
    return activation

def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"]
):
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation - proj

def plot_refusal_scores(coefs, scores, title, save_path):
    """Plot refusal scores against coefficients and save the plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(coefs, scores, marker='o', linestyle='-')
    plt.axhline(y=scores[0], color='r', linestyle='--', label='Baseline')
    plt.xlabel('Steering Coefficient')
    plt.ylabel('Refusal Score')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_name', type=str, default='gemma-2-9b-it',
        help='Name of model to extract refusal direction from')
    parser.add_argument(
        '--n_inst_test', type=int, default=100,
        help='Number of contrast pairs to compute refusal dir')
    parser.add_argument(
        '--max_tokens_generated', type=int, default=16,
        help='Maximum number of tokens to generate from model')
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size when generating completions')
    parser.add_argument(
        '--intervention_type', type=str, default='addition',
        help='Type of intervention to apply. (addition or ablation)')
    parser.add_argument(
        '--instruction_type', type=str, default='harmful',
        help='Type of instructions to use. (harmful or harmless)')
    parser.add_argument(
        '--print_completions', action='store_true',
        help='Whether to print the baseline and intervention completions')
    parser.add_argument(
        '--intervene_chat', action='store_true',
        help='Whether to intervene on chat model')
    parser.add_argument(
        '--do_coef_search', action='store_true',
        help='Whether to do a coefficient search instead of using fixed coefficients')
    parser.add_argument(
        '--coef_min', type=float, default=3,
        help='Minimum coefficient value for search')
    parser.add_argument(
        '--coef_max', type=float, default=10, 
        help='Maximum coefficient value for search')
    parser.add_argument(
        '--coef_step', type=float, default=1,
        help='Step size for coefficient search')
    parser.add_argument(
        '--use_no_template', action='store_true',
        help='Whether to use no template directions')
    parser.add_argument(
        '--use_sft', action='store_true',
        help='Whether to use SFT directions')
    parser.add_argument(
        '--sft_int', action='store_true',
        help='Whether to use Tulu directions')
    
    
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    
    assert args.intervention_type in ['addition', 'ablation', 'subtraction']
    assert args.instruction_type in ['harmful', 'harmless']
    
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()
    print(f"Number of harmful instructions: {len(harmful_inst_test)}")
    print(f"Number of harmless instructions: {len(harmless_inst_test)}")


    
    if args.model_name.startswith('gemma-2-9b'):
        model = HookedTransformer.from_pretrained_no_processing(
            args.model_name,
            device='cuda:0',
            default_padding_side='left',
            dtype=torch.float16,
        )

        model.tokenizer.padding_side = 'left'

        layer = 23
        
        if args.model_name == 'gemma-2-9b-it':
            template = GEMMA_CHAT_TEMPLATE
        elif args.model_name == 'gemma-2-9b':
            template = GEMMA_BASE_TEMPLATE
        else:
            raise ValueError(f'Unsupported model: {args.model_name}')
    
        chat_model_name = 'gemma-2-9b-it'
        base_model_name = 'gemma-2-9b'
        
        steering_coef = 20.0 if args.instruction_type == 'harmful' else 42.0
        chat_steering_coef, base_steering_coef = steering_coef, steering_coef
        
    elif args.model_name.startswith('qwen1.5-0.5b'):
        model = HookedTransformer.from_pretrained(
            args.model_name,
            default_padding_side='left',
        )

        model.tokenizer.padding_side = 'left'
        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # add padding token to tokenizer

        layer = 13

        template = QWEN_CHAT_TEMPLATE

        chat_model_name = 'qwen1.5-0.5b-chat'
        base_model_name = 'qwen1.5-0.5b'
        
        steering_coef = 2.0
        chat_steering_coef, base_steering_coef = steering_coef, steering_coef



    elif args.model_name.startswith('llama-3.1-8b'):
        
        if not args.intervene_chat:

            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
            hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16)

            model = HookedTransformer.from_pretrained(
                "meta-llama/Llama-3.1-8B",
                hf_model=hf_model,
                device="cpu",
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                tokenizer=tokenizer,
                dtype=torch.bfloat16,
                default_padding_side='left',
            )
            model.tokenizer.padding_side = 'left'
            model = model.cuda()

            template = LLAMA3_BASE_TEMPLATE
            

        else:

            if args.use_sft:
                tokenizer = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B-SFT")
                hf_model = AutoModelForCausalLM.from_pretrained("allenai/Llama-3.1-Tulu-3-8B-SFT", torch_dtype=torch.bfloat16)

                model = HookedTransformer.from_pretrained(
                    "allenai/Llama-3.1-Tulu-3-8B-SFT",
                    hf_model=hf_model,
                    device="cpu",
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    tokenizer=tokenizer,
                    dtype=torch.bfloat16,
                    default_padding_side='left',
                )
                model.tokenizer.padding_side = 'left'
                model = model.cuda()

                template = TULU_TEMPLATE
            else:

                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
                hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)

                model  = HookedTransformer.from_pretrained(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    hf_model=hf_model,
                    device="cpu",
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    tokenizer=tokenizer,
                    dtype=torch.bfloat16,
                    default_padding_side='left',
                )
                model.tokenizer.padding_side = 'left'
                model = model.cuda()

                template = LLAMA3_CHAT_TEMPLATE

        layer = 11

        if args.use_sft:
            chat_model_name = 'tulu'
        else:
            chat_model_name = 'llama-3.1-8b-instruct'
        base_model_name = 'llama-3.1-8b'
        if args.sft_int:
            base_model_name = 'tulu'
            chat_model_name = 'llama-3.1-8b-instruct'
        if args.intervention_type == 'subtraction':
            steering_coef = 2.2
        else:
            if args.intervene_chat:
                if args.use_sft:
                    steering_coef = 1.8
                else:
                    steering_coef = 2.4
            else:
                steering_coef = 1.2
        chat_steering_coef, base_steering_coef = steering_coef, steering_coef

    elif args.model_name == 'llama-2-13b':
        layer = 13
        if not args.intervene_chat:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
            hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", torch_dtype=torch.float16)

            model = HookedTransformer.from_pretrained(
                "meta-llama/Llama-2-13b-hf",
                hf_model=hf_model,
                device="cpu",
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                tokenizer=tokenizer,
                dtype=torch.float16,
                default_padding_side='left',
            )
            model.tokenizer.padding_side = 'left'
            model = model.cuda()

            template = LLAMA2_BASE_TEMPLATE
        else:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
            hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16)

            model = HookedTransformer.from_pretrained(
                "meta-llama/Llama-2-13b-chat-hf",
                hf_model=hf_model,
                device="cpu",
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                tokenizer=tokenizer,
                dtype=torch.float16,
                default_padding_side='left',
            )   
            model.tokenizer.padding_side = 'left'
            model = model.cuda()

            template = LLAMA2_CHAT_TEMPLATE
        
        base_model_name = 'llama-2-13b'
        chat_model_name = 'llama-2-13b-chat'
        
        chat_steering_coef, base_steering_coef = 5, 4

            
            
            
    elif args.model_name == 'vicuna-7b-v1.1':
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.1")
        hf_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.1", torch_dtype=torch.float16)

        model = HookedTransformer.from_pretrained(
            "llama-7b",
            hf_model=hf_model,
            device="cpu",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
            dtype=torch.float16,
            default_padding_side='left',
        )
        model.tokenizer.padding_side = 'left'
        model = model.cuda()

        layer = 14

        template = LLAMA_CHAT_TEMPLATE

        chat_model_name = 'vicuna-7b-v1.1'
        base_model_name = 'llama-7b'

        assert args.intervention_type == 'ablation', 'Only ablation is supported for vicuna-7b-v1.1'


    elif args.model_name == 'llama-7b':
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        hf_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16)

        model = HookedTransformer.from_pretrained(
            "llama-7b",
            hf_model=hf_model,
            device="cpu",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
            dtype=torch.float16,
            default_padding_side='left',
        )
        model.tokenizer.padding_side = 'left'
        model = model.cuda()

        layer = 14
    
        template = LLAMA_BASE_TEMPLATE
        
        chat_model_name = 'vicuna-7b-v1.1'
        base_model_name = 'llama-7b'
        
        chat_steering_coef = 8.0 if args.instruction_type == 'harmless' else 4.0
        base_steering_coef = 12.0
        
    else:
        raise ValueError(f'Unsupported model: {args.model_name}')
        
        
    tokenize_instructions_fn = functools.partial(tokenize_instructions, tokenizer=model.tokenizer, template=template)

    if args.use_no_template:
        base_refusal_dir = torch.load(f'refusal_directions/refusal_dir_{base_model_name}_no_template.pt')
    else:
        base_refusal_dir = torch.load(f'refusal_directions/refusal_dir_{base_model_name}.pt')
    chat_refusal_dir = torch.load(f'refusal_directions/refusal_dir_{chat_model_name}.pt')

    print(torch.cosine_similarity(chat_refusal_dir, base_refusal_dir, dim=-1))

    if args.intervention_type == 'addition':
        intervention_layers = [layer]
    elif args.intervention_type == 'subtraction':
        intervention_layers = [layer]
    else: # ablation
        intervention_layers = list(range(model.cfg.n_layers))
    
    if args.instruction_type == 'harmless':
        inst_test = harmless_inst_test
    else: # harmful
        inst_test = harmful_inst_test



    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    if args.do_coef_search:
        # Setup coefficient range
        coefs = np.arange(args.coef_min, args.coef_max + args.coef_step/2, args.coef_step).tolist()
        # Add 0 for baseline
        all_coefs = [0] + coefs
        
        # Generate baseline completions once
        print("Generating completions (baseline)")
        baseline_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=[], max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)
        baseline_score = get_refusal_scores(baseline_generations)
        
        # Initialize arrays to store results
        chat_refusal_scores = [baseline_score]
        base_refusal_scores = [baseline_score]
        
        # Perform coefficient search for chat refusal direction
        print("Performing coefficient search for chat refusal direction")
        for coef in coefs:
            print(f"Chat coefficient {coef}")
            if args.intervention_type == 'addition':
                chat_hook_fn = functools.partial(act_add_hook, direction=chat_refusal_dir, steering_coef=coef)
            elif args.intervention_type == 'subtraction':
                chat_hook_fn = functools.partial(subtract_hook, direction=chat_refusal_dir, steering_coef=coef)
            else: # ablation
                # Since ablation doesn't use coefficients, we'll just use the same hook across iterations
                chat_hook_fn = functools.partial(direction_ablation_hook, direction=chat_refusal_dir)
            
            chat_fwd_hooks = [(utils.get_act_name(act_name, l), chat_hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
            chat_intervention_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=chat_fwd_hooks, max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)
            chat_refusal_scores.append(get_refusal_scores(chat_intervention_generations))
            print(f"Chat coefficient {coef}: refusal score {chat_refusal_scores[-1]}")
        
        # Perform coefficient search for base refusal direction
        print("Performing coefficient search for base refusal direction")
        for coef in coefs:
            print(f"Base coefficient {coef}")
            if args.intervention_type == 'addition':
                base_hook_fn = functools.partial(act_add_hook, direction=base_refusal_dir, steering_coef=coef)
            elif args.intervention_type == 'subtraction':
                base_hook_fn = functools.partial(subtract_hook, direction=base_refusal_dir, steering_coef=coef)
            else: # ablation
                # Since ablation doesn't use coefficients, we'll just use the same hook across iterations
                base_hook_fn = functools.partial(direction_ablation_hook, direction=base_refusal_dir)
            
            base_fwd_hooks = [(utils.get_act_name(act_name, l), base_hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
            base_intervention_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=base_fwd_hooks, max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)
            base_refusal_scores.append(get_refusal_scores(base_intervention_generations))
            print(f"Base coefficient {coef}: refusal score {base_refusal_scores[-1]}")
        
        # Plot and save results
        intervene_chat_str = "chat" if args.intervene_chat else "base"
        chat_plot_path = f"plots/{args.model_name}_{args.instruction_type}_{args.intervention_type}_{intervene_chat_str}_chat_dir_coef_search.png"
        base_plot_path = f"plots/{args.model_name}_{args.instruction_type}_{args.intervention_type}_{intervene_chat_str}_base_dir_coef_search.png"
        
        chat_title = f"Refusal Scores for {args.model_name} with Chat Dir ({args.instruction_type}, {args.intervention_type})"
        base_title = f"Refusal Scores for {args.model_name} with Base Dir ({args.instruction_type}, {args.intervention_type})"
        
        # Create plots
        plot_refusal_scores(all_coefs, chat_refusal_scores, chat_title, chat_plot_path)
        plot_refusal_scores(all_coefs, base_refusal_scores, base_title, base_plot_path)
        
        # Print summary of findings
        print("\nCoefficient Search Results:")
        print(f"Baseline refusal score: {baseline_score}")
        
        # Find best coefficients (highest refusal score)
        if args.intervention_type == 'addition':

            #best_chat_coef_idx = np.argmax(chat_refusal_scores)
            best_base_coef_idx = np.argmax(base_refusal_scores)
        else:
            #best_chat_coef_idx = np.argmin(chat_refusal_scores)
            best_base_coef_idx = np.argmin(base_refusal_scores)
        
        if best_chat_coef_idx == 0:
            print("Chat direction: No improvement over baseline")
        else:
            best_chat_coef = all_coefs[best_chat_coef_idx]
            print(f"Chat direction: Best coefficient = {best_chat_coef}, Score = {chat_refusal_scores[best_chat_coef_idx]}")
        
        if best_base_coef_idx == 0:
            print("Base direction: No improvement over baseline")
        else:
            best_base_coef = all_coefs[best_base_coef_idx]
            print(f"Base direction: Best coefficient = {best_base_coef}, Score = {base_refusal_scores[best_base_coef_idx]}")
    
    else:
        # Original functionality - run with fixed coefficients
        print("Generating completions (no intervention)")
        baseline_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=[], max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)

        if args.intervention_type == 'addition':
            chat_hook_fn = functools.partial(act_add_hook, direction=chat_refusal_dir, steering_coef=chat_steering_coef)
        elif args.intervention_type == 'subtraction':
            chat_hook_fn = functools.partial(subtract_hook, direction=chat_refusal_dir, steering_coef=chat_steering_coef)
        else: # ablation
            chat_hook_fn = functools.partial(direction_ablation_hook, direction=chat_refusal_dir)
        
        print(f"Generating completions with chat refusal vector {args.intervention_type}")
        chat_fwd_hooks = [(utils.get_act_name(act_name, l), chat_hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
        chat_intervention_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=chat_fwd_hooks, max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)

        if args.intervention_type == 'addition':
            base_hook_fn = functools.partial(act_add_hook, direction=base_refusal_dir, steering_coef=base_steering_coef)
        else: # ablation
            base_hook_fn = functools.partial(direction_ablation_hook, direction=base_refusal_dir)
        
        print(f"Generating completions with base refusal vector {args.intervention_type}")
        base_fwd_hooks = [(utils.get_act_name(act_name, l), base_hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
        base_intervention_generations = get_generations(model, inst_test[:args.n_inst_test], tokenize_instructions_fn, fwd_hooks=base_fwd_hooks, max_tokens_generated=args.max_tokens_generated, batch_size=args.batch_size)

        if args.print_completions:
            for i in range(args.n_inst_test):
                print(f"INSTRUCTION {i}: {repr(inst_test[i])}")
                print(Fore.GREEN + f"BASELINE COMPLETION:")
                print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
                print(Fore.RED + f"INTERVENTION COMPLETION:")
                print(textwrap.fill(repr(chat_intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
                print(Fore.BLUE + f"BASE INTERVENTION COMPLETION:")
                print(textwrap.fill(repr(base_intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
                print(Fore.RESET)

        baseline_score = get_refusal_scores(baseline_generations)
        chat_score = get_refusal_scores(chat_intervention_generations)
        base_score = get_refusal_scores(base_intervention_generations)
        
        print("Baseline refusal score", baseline_score)
        print("Chat Intervention refusal score", chat_score)
        print("Base Intervention refusal score", base_score)
        
        records = {
            "model_name": args.model_name,
            "base_model_name": base_model_name,
            "chat_model_name": chat_model_name,
            "intervention": args.intervention_type,
            "dataset": args.instruction_type,
            "baseline_refusal_score": baseline_score,
            "chat_intervention_refusal_score": chat_score,
            "base_intervention_refusal_score": base_score,
            "prompt_template": template,
            "intervene_chat": args.intervene_chat,
            "use_sft": args.use_sft,
            "use_no_template": args.use_no_template
        }
        
        results_file = f"results/refusal_scores.json"
        save_records_to_json(records, results_file)