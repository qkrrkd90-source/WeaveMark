"""Downstream quality: summarization (CNN/DailyMail) and translation (WMT16
en->ro), scored with BERTScore + ROUGE/BLEU."""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessorList
from datasets import load_dataset
import evaluate

from weavemark.device import require_cuda
from weavemark.watermark import WeaveMark


def main(args):
    device = require_cuda()
    print(f"[*] Task: {args.task} | Model: {args.model_name}")

    # model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        use_safetensors=False
    ).to(device)
    model.eval()

    # dataset + metrics
    if args.task == "summarization":
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
        metric_rouge, metric_bertscore = evaluate.load("rouge"), evaluate.load("bertscore")
        max_in, max_out = 1024, 128
    else:
        dataset = load_dataset("wmt/wmt16", "ro-en", split="test", streaming=True)
        tokenizer.src_lang, tokenizer.tgt_lang = "en_XX", "ro_RO"
        metric_bleu, metric_bertscore = evaluate.load("sacrebleu"), evaluate.load("bertscore")
        max_in, max_out = 128, 128

    logits_processor = LogitsProcessorList()
    if not args.no_watermark:
        wp = WeaveMark(
            tokenizer=tokenizer, device=device, vocab_size=tokenizer.vocab_size,
            top_k=args.top_k, delta=args.delta, window_size=args.window_size,
            bits=args.bits, bits_per_token=args.bpt,
            partition_seeds=list(range(args.num_layers)),
            # zero-bit parameters
            enable_zerobit=args.enable_zerobit,
            num_zerobit_layers=args.num_zerobit_layers,
            layer_shuffle=not args.no_layer_shuffle,
            zerobit_partition_key=args.zerobit_partition_key,
            zerobit_c_key=args.zerobit_c_key
        )
        logits_processor.append(wp)

    predictions, references = [], []
    dataset_iter = iter(dataset)

    for _ in tqdm(range(args.num_samples), desc="Generating"):
        try:
            item = next(dataset_iter)
        except StopIteration:
            break

        source = item["article"] if args.task == "summarization" else item["translation"]["en"]
        target = item["highlights"] if args.task == "summarization" else item["translation"]["ro"]
        inputs = tokenizer(source, return_tensors="pt", max_length=max_in, truncation=True).to(device)

        with torch.no_grad():
            gen_config = {
                "max_new_tokens": max_out,
                "logits_processor": logits_processor,
                "do_sample": args.do_sample,
            }
            if args.do_sample:
                gen_config.update({"temperature": args.temperature, "top_k": args.top_k})
            else:
                gen_config["num_beams"] = 4

            if args.task == "summarization":
                gen_config.update({"min_length": 56, "length_penalty": 2.0, "no_repeat_ngram_size": 3})
            else:
                gen_config["forced_bos_token_id"] = tokenizer.lang_code_to_id["ro_RO"]

            outputs = model.generate(**inputs, **gen_config)

        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        references.append(target)

    # metrics
    print("\n[*] Calculating metrics...")
    actual_rescale = args.use_rescale if args.task == "summarization" else False

    safe_predictions = [" ".join(p.split()[:250]) for p in predictions]
    safe_references = [" ".join(r.split()[:250]) for r in references]

    try:
        bert_res = metric_bertscore.compute(
            predictions=safe_predictions,
            references=safe_references,
            lang="ro" if args.task == "translation" else "en",
            model_type=args.scorer_model,
            rescale_with_baseline=actual_rescale,
            device=device,
            batch_size=8
        )
        bert_scores = np.array(bert_res["f1"]) * 100
        results = {
            "BERTScore_F1_mean": np.mean(bert_scores),
            "BERTScore_F1_std": np.std(bert_scores)
        }
        print("[*] BERTScore computed.")

    except Exception as e:
        print(f"\n[!] BERTScore computation failed: {e}")
        results = {"BERTScore_F1_mean": 0.0, "BERTScore_F1_std": 0.0}

    if args.task == "summarization":
        # per-sample scores
        rouge_res = metric_rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
            use_aggregator=False
        )
        for k, v in rouge_res.items():
            scores = np.array(v) * 100
            results[f"ROUGE-{k[-1].upper()}_mean"] = np.mean(scores)
            results[f"ROUGE-{k[-1].upper()}_std"] = np.std(scores)
    else:
        # corpus BLEU (mean)
        results["BLEU_mean"] = metric_bleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )["score"]

        # per-sentence BLEU (std)
        sentence_bleus = []
        for p, r in zip(predictions, references):
            safe_p = p if p.strip() else "empty"
            score = metric_bleu.compute(predictions=[safe_p], references=[[r]])["score"]
            sentence_bleus.append(score)
        results["BLEU_std"] = np.std(sentence_bleus)

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "no_wm" if args.no_watermark else f"del{args.delta}"
    suffix += f"_w{args.window_size}"
    with open(os.path.join(args.output_dir, f"res_{args.task}_{suffix}.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"[*] Results saved in {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["summarization", "translation"])
    p.add_argument("--model_name", required=True)
    p.add_argument("--scorer_model", default="microsoft/deberta-large-mnli")
    p.add_argument("--use_rescale", action="store_true")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--output_dir", default="output_quality")
    p.add_argument("--no_watermark", action="store_true")
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--bpt", type=int, default=1)
    p.add_argument("--num_layers", type=int, default=10)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--window_size", type=int, default=2)
    p.add_argument("--bits", default="1010101010")
    # zero-bit parameters
    p.add_argument("--enable_zerobit", action="store_true")
    p.add_argument("--num_zerobit_layers", type=int, default=10)
    p.add_argument("--no_layer_shuffle", action="store_true")
    p.add_argument("--zerobit_partition_key", type=int, default=998877)
    p.add_argument("--zerobit_c_key", type=int, default=665544)
    main(p.parse_args())
