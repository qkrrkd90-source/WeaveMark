"""Generate watermarked (or plain) text into output_dump/<run_folder>/."""
import argparse
import json
import os
import time
from datetime import datetime
from random import randint

# set before torch inits CUDA; setdefault lets CUDA_VISIBLE_DEVICES override
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from weavemark.data import record_data, load_data
from weavemark.device import require_cuda
from weavemark.watermark import WeaveMark
from weavemark.ecc import Golay24, DualGolay48, ReedMuller32_16, DualRM64


def save_human_written_as_json(prompt_idx, prompts, human_written, save_dir):
    json_data = [{"prompt_idx": idx, "prompt": p, "human_completion": c}
                 for idx, p, c in zip(prompt_idx, prompts, human_written)]
    with open(os.path.join(save_dir, "human_written.jsonl"), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def append_runtime(save_dir, runtime):
    params_path = os.path.join(save_dir, "generation_params.json")
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    params['runtime'] = runtime
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)


def create_save_dir(args, method, time_str):
    ecc_str = f"_{args.ecc_method}" if args.method.lower() == 'weavemark' else ""
    folder_name = f"{method}{ecc_str}_{args.dataset.replace('/', '_')}_{args.model_name.replace('/', '-')}_{time_str}"
    save_dir = os.path.join("output_dump", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def main(args):
    require_cuda()
    args.partition_seeds = list(range(args.num_layers))

    begin = time.time()
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_').replace(":", "-")
    save_dir = create_save_dir(args, args.method, time_str)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        offload_folder="offload",
        attn_implementation="sdpa",  # SDPA (Flash Attention fallback)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model.eval()
    prompt_idx, prompts, human_written, num_test = load_data(
        args.dataset, args.prompt_len, args.num_test,
        model_name=args.model_name, opengen_path=args.opengen_path)

    with open(os.path.join(save_dir, "prompt_idx.txt"), "w") as f:
        for idx in prompt_idx:
            f.write(str(idx) + '\n')
    save_human_written_as_json(prompt_idx, prompts, human_written, save_dir)

    vocab_size = model.config.vocab_size
    batch_size = args.batch_size
    n_batches = int(np.ceil(num_test / batch_size))
    pbar = tqdm(total=n_batches)

    for batch in range(n_batches):
        batch_prompts = prompts[batch * batch_size: min(num_test, (batch + 1) * batch_size)]
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        prompt_tokens_len = input_ids.size(-1)

        if args.method.lower() == 'weavemark':
            if args.random_message:
                ecc_type = args.ecc_method.lower()
                if ecc_type == 'golay':
                    golay = Golay24(); msg_bits = np.random.randint(0, 2, 12).tolist()
                    encoded_bits = golay.encode(msg_bits)
                elif ecc_type == 'dual_golay':
                    dual_golay = DualGolay48(); msg_bits = np.random.randint(0, 2, 24).tolist()
                    encoded_bits = dual_golay.encode(msg_bits)
                elif ecc_type == 'rm':
                    rm = ReedMuller32_16(); msg_bits = np.random.randint(0, 2, 16).tolist()
                    encoded_bits = rm.encode(msg_bits)
                elif ecc_type == 'dual_rm':
                    dual_rm = DualRM64(); msg_bits = np.random.randint(0, 2, 32).tolist()
                    encoded_bits = dual_rm.encode(msg_bits)
                else:
                    msg_bits = [randint(0, 1) for _ in range(len(args.message))]
                    encoded_bits = msg_bits

                bits = "".join(map(str, encoded_bits))
                orig_msg_str = "".join(map(str, msg_bits))
            else:
                bits = args.message
                orig_msg_str = "unknown"

            watermark_processor = WeaveMark(
                tokenizer=tokenizer, vocab_size=vocab_size, device=model.device, top_k=args.top_k,
                partition_seeds=args.partition_seeds, c_key=args.c_key, bit_idx_key=args.bit_idx_key,
                delta=args.prob_delta, window_size=args.window_size, bits=bits,
                bits_per_token=args.bpt,
                layer_shuffle=not args.no_layer_shuffle,
                # zero-bit parameters
                enable_zerobit=args.enable_zerobit,
                num_zerobit_layers=args.num_zerobit_layers,
                zerobit_partition_key=args.zerobit_partition_key,
                zerobit_c_key=args.zerobit_c_key
            )

            generate_args = {
                'logits_processor': [watermark_processor],
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature,
                'attention_mask': attention_mask,
                'do_sample': args.do_sample,
                'top_k': args.top_k
            }
        elif args.method.lower() == 'no_watermark':
            generate_args_null = {'max_new_tokens': args.max_new_tokens, 'attention_mask': attention_mask,
                                  'temperature': args.temperature, 'do_sample': True, 'top_k': args.top_k}

        idx_list = [prompt_idx[i] for i in range(batch * batch_size, min(num_test, (batch + 1) * batch_size))]

        if args.method.lower() == 'no_watermark':
            with torch.inference_mode():
                null_token = model.generate(input_ids, **generate_args_null)

            # codeword length for the ECC method
            ecc_bits_len_map = {
                'golay': 24, 'dual_golay': 48, 'rm': 32, 'dual_rm': 64,
                'none': len(args.message)
            }
            bits_len = ecc_bits_len_map.get(args.ecc_method.lower(), len(args.message))

            params = {
                "method": "no_watermark",
                "model_name": args.model_name,
                "vocab_size": vocab_size,
                "temperature": args.temperature,
                "max_length": args.max_new_tokens,
                "window_size": args.window_size,
                # detection parameters (used for FPR measurement)
                "ecc_method": args.ecc_method,
                "bits_len": bits_len,
                "bpt": args.bpt,
                "num_layers": args.num_layers,
                "prob_delta": args.prob_delta,
                "partition_seeds": list(range(args.num_layers)),
                "c_key": args.c_key,
                "bit_idx_key": args.bit_idx_key,
                # zero-bit parameters
                "layer_shuffle": not args.no_layer_shuffle,
                "enable_zerobit": args.enable_zerobit,
                "num_zerobit_layers": args.num_zerobit_layers,
                "zerobit_partition_key": args.zerobit_partition_key,
                "zerobit_c_key": args.zerobit_c_key
            }
            record_data(batch_prompts, tokenizer, null_token[:, prompt_tokens_len:].tolist(), idx_list, save_dir, params)

        if args.method.lower() == 'weavemark':
            with torch.inference_mode():
                new_token = model.generate(input_ids, **generate_args)

            params = {
                "method": "weavemark", "ecc_method": args.ecc_method, "orig_msg": orig_msg_str, "codeword": bits,
                "bpt": args.bpt, "num_layers": args.num_layers,
                "model_name": args.model_name, "vocab_size": vocab_size, "bits_len": len(bits),
                "top_k": args.top_k, "do_sample": args.do_sample, 'temperature': args.temperature,
                "max_length": args.max_new_tokens, "prob_delta": args.prob_delta, "window_size": args.window_size,
                "c_key": args.c_key, "bit_idx_key": args.bit_idx_key, "partition_seeds": args.partition_seeds,
                "time_stamp": time_str,
                # zero-bit parameters
                "layer_shuffle": not args.no_layer_shuffle,
                "enable_zerobit": args.enable_zerobit,
                "num_zerobit_layers": args.num_zerobit_layers,
                "zerobit_partition_key": args.zerobit_partition_key,
                "zerobit_c_key": args.zerobit_c_key
            }
            record_data(batch_prompts, tokenizer, new_token[:, prompt_tokens_len:].tolist(), idx_list, save_dir, params, bits=bits)
        pbar.update(1)

    end = time.time()
    runtime = end - begin
    append_runtime(save_dir, runtime)


if __name__ == "__main__":
    def list_of_ints(arg):
        return arg if type(arg) is list else list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--method", type=str, default='weavemark', choices=["weavemark", "no_watermark"])
    parser.add_argument("--ecc_method", type=str, default="none",
                        choices=["none", "golay", "dual_golay", "rm", "dual_rm"])
    parser.add_argument("--bpt", type=int, default=5, help="bits per token (kappa)")
    parser.add_argument("--num_layers", type=int, default=10, help="number of multi-bit layers (ell)")
    parser.add_argument("--prob_delta", type=float, default=1.0)
    parser.add_argument("--partition_seeds", type=list_of_ints, default=list(range(10)))
    parser.add_argument("--c_key", type=int, default=8214793)
    parser.add_argument("--bit_idx_key", type=int, default=283519)
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--prompt_len", type=int, default=100)
    parser.add_argument("--opengen_path", type=str, default=os.path.join("data", "OpenGen.jsonl"),
                        help="Local OpenGen JSONL (used when --dataset opengen)")
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--message", type=str, default="0" * 16)
    parser.add_argument("--random_message", action='store_true')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action='store_true')
    # zero-bit parameters
    parser.add_argument("--no_layer_shuffle", action='store_true',
                        help="Disable layer->bit spreading (ablation; default is enabled)")
    parser.add_argument("--enable_zerobit", action='store_true')
    parser.add_argument("--num_zerobit_layers", type=int, default=1)
    parser.add_argument("--zerobit_partition_key", type=int, default=998877)
    parser.add_argument("--zerobit_c_key", type=int, default=665544)
    args = parser.parse_args()
    main(args)