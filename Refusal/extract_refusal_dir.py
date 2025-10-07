import torch
import functools
import gc
import argparse

from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_harmful_instructions, get_harmless_instructions, tokenize_instructions, GEMMA_CHAT_TEMPLATE, GEMMA_BASE_TEMPLATE, QWEN_CHAT_TEMPLATE, LLAMA_CHAT_TEMPLATE, LLAMA_BASE_TEMPLATE, LLAMA3_CHAT_TEMPLATE, LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM, LLAMA3_BASE_TEMPLATE, NO_TEMPLATE, MISTRAL_TEMPLATE, LLAMA2_CHAT_TEMPLATE, LLAMA2_BASE_TEMPLATE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_name', type=str, default='gemma-2-9b-it',
        help='Name of model to extract refusal direction from')
    parser.add_argument(
        '--n_inst_train', type=int, default=128,
        help='Number of contrast pairs to compute refusal dir')
    
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()
    
    if args.model_name.startswith('gemma-2-9b'):
        model = HookedTransformer.from_pretrained_no_processing(
            args.model_name,
            device='cuda:0',
            default_padding_side='left',
            dtype=torch.float16,
        )

        model.tokenizer.padding_side = 'left'

        pos = -1
        layer = 23
        
        if args.model_name == 'gemma-2-9b-it':
            template = GEMMA_CHAT_TEMPLATE
        elif args.model_name == 'gemma-2-9b':
            template = GEMMA_BASE_TEMPLATE
        else:
            raise ValueError(f'Unsupported model: {args.model_name}')
        
    elif args.model_name.startswith('qwen1.5-0.5b'):
        model = HookedTransformer.from_pretrained(
            args.model_name,
            default_padding_side='left',
        )

        model.tokenizer.padding_side = 'left'
        model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # add padding token to tokenizer

        pos = -1
        layer = 13
    
        template = QWEN_CHAT_TEMPLATE
    
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

        pos = -1
        layer = 14

        template = LLAMA_CHAT_TEMPLATE

    elif args.model_name == 'llama-7b':
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        hf_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.bfloat16)

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

        pos = -1
        layer = 14
        
        template = LLAMA_BASE_TEMPLATE

    elif args.model_name == 'llama-3.1-8b':
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

        pos = -4
        layer = 11
        
        template = LLAMA3_BASE_TEMPLATE
    
    elif args.model_name == 'llama-3.1-8b-instruct':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)

        model = HookedTransformer.from_pretrained(
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

        pos = -2
        layer = 11
        
        template = LLAMA3_CHAT_TEMPLATE

    elif args.model_name == 'tulu':
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

        pos = -1
        layer = 11
        
        template = LLAMA3_BASE_TEMPLATE

    elif args.model_name == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        hf_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", torch_dtype=torch.bfloat16)

        model = HookedTransformer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
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

        pos = -1
        layer = 12
        
        template = MISTRAL_TEMPLATE

    elif args.model_name == 'llama-2-13b':  
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

        pos = -4
        layer = 14

        template = LLAMA2_BASE_TEMPLATE

    elif args.model_name == 'llama-2-13b-chat':
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

        pos = -1
        layer = 14

        template = LLAMA2_CHAT_TEMPLATE
    else:
        raise ValueError(f'Unsupported model: {args.model_name}')
        
        
    tokenize_instructions_fn = functools.partial(tokenize_instructions, tokenizer=model.tokenizer, template=template)

    print(f"Using {args.n_inst_train} pairs to compute refusal direction")
    harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[:args.n_inst_train])
    harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[:args.n_inst_train])

    harmful_logits, harmful_cache = model.run_with_cache(harmful_toks, names_filter=utils.get_act_name('resid_pre', layer))
    harmless_logits, harmless_cache = model.run_with_cache(harmless_toks, names_filter=utils.get_act_name('resid_pre', layer))

    #save harmful_cache['resid_pre', layer][:, pos, :] to file
    torch.save(harmful_cache['resid_pre', layer][:, pos, :], f'refusal_directions/harmful_acts_{args.model_name}.pt')
    torch.save(harmless_cache['resid_pre', layer][:, pos, :], f'refusal_directions/harmless_acts_{args.model_name}.pt')


    harmful_mean_act = harmful_cache['resid_pre', layer][:, pos, :].mean(dim=0)
    harmless_mean_act = harmless_cache['resid_pre', layer][:, pos, :].mean(dim=0)

    refusal_dir = harmful_mean_act - harmless_mean_act
    refusal_dir = refusal_dir / refusal_dir.norm()

    # clean up memory
    del harmful_cache, harmless_cache, harmful_logits, harmless_logits
    gc.collect(); torch.cuda.empty_cache()
    if template != NO_TEMPLATE:
        save_file = f'refusal_directions/refusal_dir_{args.model_name}.pt'
    else:
        save_file = f'refusal_directions/refusal_dir_{args.model_name}_no_template.pt'
    torch.save(refusal_dir, save_file)
    print(f'Saved refusal direction to {save_file}')