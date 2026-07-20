"""Detection: multi-bit extraction, zero-bit z-score, PPL. Reads a generate.py
run folder and writes per-sample and summary CSVs."""
import argparse
import json
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from weavemark.device import require_cuda
from weavemark.extraction import extract_bits, detect_zerobit
from weavemark.ecc import (
    Golay24, DualGolay48, ReedMuller32_16, DualRM64,
    build_golay_soft_decoder, build_dual_golay_soft_decoder,
    build_rm_soft_decoder, build_dual_rm_soft_decoder,
)
from weavemark.watermark import WeaveMark


def main(args):
    device = require_cuda()

    params_path = os.path.join(args.data_dir, "generation_params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # no_watermark (FPR)
    is_no_watermark = params.get("method", "weavemark") == "no_watermark"

    # zero-bit only
    is_zerobit_only = params.get("num_layers", 10) == 0

    ecc_method = params.get("ecc_method", "none")
    bits_len = params.get("bits_len", 12)
    model_name = params.get("model_name", "meta-llama/Meta-Llama-3-8B")

    bpt = params.get("bpt", 1)
    num_layers = params.get("num_layers", bits_len)

    # zero-bit params
    layer_shuffle = params.get("layer_shuffle", True)
    enable_zerobit = params.get("enable_zerobit", True)
    num_zerobit_layers = params.get("num_zerobit_layers", 1)
    zerobit_partition_key = params.get("zerobit_partition_key", 998877)
    zerobit_c_key = params.get("zerobit_c_key", 665544)

    # z_score always emitted when enable_zerobit; z_success only with a threshold
    run_zerobit_detail = enable_zerobit
    run_zerobit_summary = args.zerobit_threshold is not None and enable_zerobit

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token

    text_file_path = os.path.join(args.data_dir, args.text_file)
    data = []
    with open(text_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # same extractor (also used on no_watermark text)
    watermark_processor = WeaveMark(
        tokenizer=tokenizer, vocab_size=params["vocab_size"], device=device,
        partition_seeds=params.get("partition_seeds", list(range(num_layers))),
        c_key=params.get("c_key", 530773),
        bit_idx_key=params.get("bit_idx_key", 283519),
        delta=params.get("prob_delta", 1.0),
        window_size=params.get("window_size", 2),
        bits="0" * bits_len,
        bits_per_token=bpt,
        layer_shuffle=layer_shuffle,
        enable_zerobit=enable_zerobit,
        zerobit_partition_key=zerobit_partition_key,
        zerobit_c_key=zerobit_c_key
    )

    decoder_builder = None
    if ecc_method == "golay":
        decoder_builder = build_golay_soft_decoder(Golay24())
    elif ecc_method == "dual_golay":
        decoder_builder = build_dual_golay_soft_decoder(DualGolay48())
    elif ecc_method == "rm":
        decoder_builder = build_rm_soft_decoder(ReedMuller32_16())
    elif ecc_method == "dual_rm":
        decoder_builder = build_dual_rm_soft_decoder(DualRM64())

    scorer = None
    if args.perplexity:
        from perplexity import LocalModel
        scorer = LocalModel(model_name=args.ppl_model_name, device=device)

    evaluation_results = []

    for item in tqdm(data, desc="Evaluating"):
        prompt = item.get("prompt", "")
        text = item.get("attacked_text") if "attacked_text" in item else item.get("generation_text", "")

        if not text or not str(text).strip():
            continue

        codeword_str = item.get("bits", params.get("codeword", ""))

        orig_msg_bits = []
        if decoder_builder and codeword_str:  # only decode when a codeword is present
            polar_codeword = [1 if b == '1' else -1 for b in codeword_str]
            orig_msg_bits = decoder_builder(polar_codeword)

        if args.detect or args.perplexity:
            llama_tokens = tokenizer.encode(text, add_special_tokens=False)
            actual_len = len(llama_tokens)

            cropped_len = item.get('cropped_length', None)

            check_lengths = list(range(args.stride, actual_len + 1, args.stride))
            if not check_lengths:
                check_lengths = [actual_len]

            for check_len in check_lengths:
                if args.stride == 999 and cropped_len is not None:
                    sliced_llama_tokens = llama_tokens
                    display_len = cropped_len
                else:
                    sliced_llama_tokens = llama_tokens[:check_len]
                    display_len = check_len

                if len(sliced_llama_tokens) <= params["window_size"]:
                    continue

                sliced_text = tokenizer.decode(sliced_llama_tokens, skip_special_tokens=True)

                ppl_val = None
                if args.perplexity and sliced_text.strip():
                    if display_len <= args.ppl_length:
                        p = scorer.get_perplexity(prompt, [sliced_text])[0]
                        if p > 0:
                            ppl_val = p

                # zero-bit z-score
                z_score, green_count, total_count, p_value, z_success = None, None, None, None, None

                if args.detect and run_zerobit_detail:
                    zerobit_result = detect_zerobit(
                        tokens=sliced_llama_tokens,
                        vocab_size=params["vocab_size"],
                        window_size=params.get("window_size", 2),
                        zerobit_partition_key=zerobit_partition_key,
                        zerobit_c_key=zerobit_c_key,
                        num_zerobit_layers=num_zerobit_layers
                    )
                    z_score = zerobit_result['z_score']
                    green_count = zerobit_result['green_count']
                    total_count = zerobit_result['total_count']
                    p_value = zerobit_result['p_value']
                    if args.zerobit_threshold is not None:
                        z_success = z_score > args.zerobit_threshold

                # multi-bit extraction
                accuracy, success, raw_match_rate = 0.0, False, 0.0
                extracted_str, decoded_msg_str = "", ""

                if args.detect and not is_no_watermark and not is_zerobit_only:
                    extracted_str, decoder_input, bit_counts = extract_bits(
                        tokens=sliced_llama_tokens,
                        processor=watermark_processor,
                        codeword_len=len(codeword_str),
                        k=bpt,
                        num_layers=num_layers,
                        hard_decision=args.hard_decision,
                        use_llr=args.use_llr
                    )

                    raw_match_rate = sum([1 for a, b in zip(codeword_str, extracted_str) if a == b]) / len(codeword_str)

                    if decoder_builder:
                        decoded_msg_bits = decoder_builder(decoder_input)
                        decoded_msg_str = "".join(map(str, decoded_msg_bits))
                        match_count = sum([1 for a, b in zip(orig_msg_bits, decoded_msg_bits) if a == b])
                        accuracy = match_count / len(orig_msg_bits)
                        success = (match_count == len(orig_msg_bits))
                    else:
                        accuracy, success = raw_match_rate, (raw_match_rate == 1.0)

                interval_str = f"0-{display_len}"

                result_dict = {
                    "prompt_idx": item.get("prompt_idx", 0),
                    "length": display_len,
                    "interval": interval_str,
                    "eval_length": len(sliced_llama_tokens)
                }

                if args.detect:
                    # multi-bit results
                    if not is_no_watermark and not is_zerobit_only:
                        result_dict.update({
                            "raw_match_rate": raw_match_rate,
                            "accuracy": accuracy,
                            "success": success,
                            "extracted_raw": extracted_str,
                            "decoded_msg": decoded_msg_str
                        })
                    # zero-bit results
                    if run_zerobit_detail:
                        result_dict.update({
                            "total_count": total_count,
                            "green_count": green_count,
                            "z_score": z_score
                        })
                        if args.zerobit_threshold is not None:
                            result_dict["z_success"] = z_success

                if args.perplexity:
                    result_dict["ppl"] = ppl_val

                evaluation_results.append(result_dict)

    if evaluation_results:
        df_results = pd.DataFrame(evaluation_results)

        detailed_save_name = f"detailed_eval_{args.text_file.replace('.jsonl', '')}.csv"
        df_results.to_csv(os.path.join(args.data_dir, detailed_save_name), index=False)

        agg_dict = {
            'interval': ('interval', 'first'),
            'sample_count': ('prompt_idx', 'count')
        }
        if args.detect:
            if not is_no_watermark and not is_zerobit_only:
                agg_dict.update({
                    'raw_match_rate_mean': ('raw_match_rate', 'mean'),
                    'accuracy_mean': ('accuracy', 'mean'),
                    'success_rate': ('success', 'mean')
                })
            if run_zerobit_summary:
                agg_dict.update({
                    'z_score_mean': ('z_score', 'mean'),
                    'z_score_std': ('z_score', 'std'),
                    'z_success_rate': ('z_success', 'mean'),
                    'green_count_mean': ('green_count', 'mean'),
                    'total_count_mean': ('total_count', 'mean')
                })
        if args.perplexity:
            agg_dict.update({
                'ppl_mean': ('ppl', 'mean')
            })

        summary_df = df_results.groupby('length').agg(**agg_dict).reset_index()

        summary_save_name = f"summary_eval_{args.text_file.replace('.jsonl', '')}.csv"
        summary_df.to_csv(os.path.join(args.data_dir, summary_save_name), index=False)

        print("\n=== [Per prefix-length evaluation summary] ===")
        print(summary_df.to_string(index=False))
        print(f"\n[*] Detailed results: {detailed_save_name}")
        print(f"[*] Summary results:  {summary_save_name}")
    else:
        print("\n[warning] No valid text found; nothing was evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECC watermark detection and evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder containing the generated data")
    parser.add_argument("--text_file", type=str, default="generation_text.jsonl")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--hard_decision", action="store_true", help="Hard-decision voting")
    parser.add_argument("--use_llr", action="store_true", help="Use log-likelihood ratio")
    parser.add_argument("--zerobit_threshold", type=float, default=None,
                        help="Zero-bit z-score threshold (e.g. 4.0). If unset, z-score detection is disabled.")
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--ppl_length", type=int, default=100)
    parser.add_argument("--ppl_model_name", type=str, default="google/gemma-2-9b")
    args = parser.parse_args()
    main(args)
