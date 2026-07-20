"""DIPPER paraphrase attack (crop-then-attack): crop each text, paraphrase each
crop with DIPPER. One output file per (lexical, order) setting."""
import argparse
import os
import json
import gc

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from dipper import DipperParaphraser


def main(args):
    # DIPPER (T5-XXL, 4-bit)
    dp = DipperParaphraser(
        model=args.dipper_model,
        quantize_4bit=(not args.no_quantize),
        verbose=True,
    )

    # generator tokenizer for cropping
    print(f"[DIPPER] Loading {args.model_name} tokenizer for token cropping")
    llama_tok = AutoTokenizer.from_pretrained(args.model_name)

    # input data
    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # length sweep (short: DIPPER is slow)
    target_lengths = [200]

    # build crops
    jobs = []  # (item_idx, length, cropped_text)
    for item_idx, item in enumerate(data):
        original_text = item.get('generation_text', '')
        if not original_text.strip():
            continue
        tokens = llama_tok.encode(original_text, add_special_tokens=False)
        for length in target_lengths:
            if len(tokens) < length:
                continue
            cropped_tokens = tokens[:length]
            cropped_text = llama_tok.decode(cropped_tokens, skip_special_tokens=True)
            jobs.append((item_idx, length, cropped_text))

    # output file
    output_file = f"dipper_text_L{args.lex_diversity}_O{args.order_diversity}.jsonl"
    output_path = os.path.join(args.data_dir, output_file)

    # DIPPER defaults
    gen_kwargs = dict(
        do_sample=True,
        top_p=0.75,
        top_k=0,  # 0 disables top-k in T5 generate (None is not accepted)
        max_length=args.max_length,
    )

    # paraphrase in batches
    desc = f"DIPPER L{args.lex_diversity}_O{args.order_diversity}"
    with open(output_path, 'w', encoding='utf-8') as f:
        for batch_start in tqdm(range(0, len(jobs), args.batch_size), desc=desc):
            batch = jobs[batch_start: batch_start + args.batch_size]
            texts = [j[2] for j in batch]
            prefixes = [data[j[0]].get('prompt', '') for j in batch]

            try:
                paraphrased = dp.paraphrase_batch(
                    texts,
                    lex_diversity=args.lex_diversity,
                    order_diversity=args.order_diversity,
                    prefixes=prefixes,
                    sent_interval=args.sent_interval,
                    **gen_kwargs,
                )
            except Exception as e:
                print(f"[WARN] batch failed at {batch_start}: {e}. Filling empty.")
                paraphrased = ["" for _ in batch]

            for (item_idx, length, cropped_text), para_text in zip(batch, paraphrased):
                item = data[item_idx]
                new_item = dict(item)
                new_item['cropped_length'] = length
                new_item['attacked_text'] = para_text
                new_item['original_cropped_text'] = cropped_text
                new_item['lex_diversity'] = args.lex_diversity
                new_item['order_diversity'] = args.order_diversity
                f.write(json.dumps(new_item, ensure_ascii=False) + '\n')

            # free VRAM
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[DIPPER] Done -> {output_file}  ({len(jobs)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, default="generation_text.jsonl")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Generator tokenizer (for token-level cropping)")
    parser.add_argument("--dipper_model", type=str,
                        default="kalpeshk2011/dipper-paraphraser-xxl")
    parser.add_argument("--lex_diversity", type=int, default=60,
                        choices=[0, 20, 40, 60, 80, 100])
    parser.add_argument("--order_diversity", type=int, default=0,
                        choices=[0, 20, 40, 60, 80, 100])
    parser.add_argument("--batch_size", type=int, default=4,
                        help="DIPPER chunk batch size (reduce on OOM)")
    parser.add_argument("--sent_interval", type=int, default=3,
                        help="Sentences per paraphrase chunk (DIPPER default = 3)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--no_quantize", action="store_true",
                        help="Disable 4-bit NF4 quantization (>=24GB VRAM recommended)")
    args = parser.parse_args()
    main(args)
