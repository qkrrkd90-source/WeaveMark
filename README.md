# WeaveMark

Reference implementation of *Robust and Scalable Multi-bit LLM Watermarking via
Coded Payload Spreading*.

WeaveMark embeds a *k*-bit message into LLM-generated text while keeping the
token distribution unbiased. It combines coded payload spreading (κ codeword
bits per token over ℓ layers, with a context-shuffled layer→bit assignment),
soft-decision ECC decoding from per-bit vote margins, and unbiased multilayer
reweighting (each direction XORed with a context coin flip). Dedicated zero-bit
layers, partitioned from the context rather than the message, give
message-independent presence detection.

## Layout

```
weavemark/            library
  watermark.py        WeaveMark logits processor (embedding)
  extraction.py       extract_bits, detect_zerobit
  prf.py              keyed PRF (shared by embed/extract)
  data.py             dataset loading, prompt prep, JSONL records
  device.py           CUDA check
  ecc/                Golay, Reed-Muller, soft-decision decoders
generate.py           generate watermarked / plain text
detect.py             extraction + zero-bit z-score + PPL
evaluate_downstream.py  summarization / translation quality
perplexity.py         PPL scorer
attacks/              substitution, DIPPER
run_*.py              pipelines (generate+detect, attacks, downstream)
data/OpenGen.jsonl    OpenGen prompts (--dataset opengen); see note below
```

`data/OpenGen.jsonl` (the standard public OpenGen benchmark) is omitted from
this archive for size; place it under `data/` to use `--dataset opengen`. The
`c4` and `openwebtext` datasets stream from the Hub and need no local file.

Run all scripts from the repository root so `import weavemark` resolves.

## Install

Requires Python 3.10–3.12 and a CUDA GPU (models load 4-bit via bitsandbytes;
there is no CPU path). 4-bit LLaMA-3-8B fits in ~16 GB.

```bash
cd WeaveMark
python -m venv .venv && source .venv/bin/activate     # or: conda create -n weavemark python=3.11
```

Install a CUDA build of PyTorch **before** `requirements.txt` — plain
`pip install torch` gives a CPU-only wheel on Windows, and the wheel must have
kernels for your GPU (RTX 50-series / sm_120 needs torch ≥ 2.7 on CUDA ≥ 12.8;
older cu121 wheels stop at sm_90 and fail at runtime). Pick the matching command
from https://pytorch.org.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # example: Blackwell
pip install -r requirements.txt
```

Check the GPU actually runs kernels (`is_available()` alone is not enough — it is
True even when the wheel lacks kernels for the arch):

```bash
python -c "import torch; t=torch.zeros(8).cuda().normal_(); print(torch.__version__, torch.cuda.get_device_capability(), (t*t).sum().item())"
```

LLaMA-3 and C4 are gated: `huggingface-cli login` and accept their terms.
OpenGen ships with the repo. Select a GPU with `CUDA_VISIBLE_DEVICES=<i>`.

## Quickstart

```bash
python generate.py --method weavemark --ecc_method rm --random_message \
  --dataset c4 --num_test 100 --max_new_tokens 200 \
  --bpt 10 --num_layers 10 --window_size 2 --prob_delta 1.0 \
  --num_zerobit_layers 0 --do_sample

python detect.py --data_dir output_dump/<run_folder> --detect --stride 25
```

`detect.py` writes `detailed_eval_*.csv` and `summary_eval_*.csv`; key columns
are `success_rate` and, with `--zerobit_threshold`, `z_success_rate`.
`run_main.py` chains generate + detect (edit the constants at its top).

Datasets: `c4` / `openwebtext` stream from the Hub and truncate to
`--prompt_len` words; `opengen` reads `data/OpenGen.jsonl` and truncates each
prefix to `--prompt_len` tokens.

## Parameters

| Flag | Symbol | Meaning | Typical |
|------|--------|---------|---------|
| `--bpt` | κ | bits per token | 10 |
| `--num_layers` | ℓ | multi-bit layers | 10 |
| `--window_size` | h | context window | 2 |
| `--prob_delta` | δ | reweighting strength | 1.0 |
| `--ecc_method` | — | `none`/`golay`/`dual_golay`/`rm`/`dual_rm` | `rm` |
| `--num_zerobit_layers` | ℓ_z | zero-bit layers | 0 or 2 |
| `--top_k` | K | sampling top-k | 50 |

ECC msg→codeword: `golay` 12→24, `rm` 16→32, `dual_golay` 24→48, `dual_rm`
32→64. `none` embeds `--message` uncoded.

Paper settings: multi-bit tracing uses `--bpt 10 --num_layers 10 --window_size 2
--prob_delta 1.0 --num_zerobit_layers 0`, ECC by message length; add
`--num_zerobit_layers 2 --enable_zerobit` for the combined setting; pure zero-bit
is `--bpt 0 --num_layers 0 --num_zerobit_layers 10 --enable_zerobit`.

## Attacks and downstream

```bash
python attacks/substitution.py --data_dir output_dump/<run> --ratios 0.1 0.2 0.3
python attacks/dipper_attack.py --data_dir output_dump/<run> --lex_diversity 20 --order_diversity 20
python detect.py --data_dir output_dump/<run> --text_file attacked_text_10.jsonl --detect --stride 999
```

`run_substitution.py` / `run_dipper.py` chain attack + whole-text detection
(`--stride 999` evaluates each crop whole). `run_downstream.py` runs
summarization (CNN/DailyMail) and translation (WMT16 en→ro) with BERTScore +
ROUGE/BLEU.

## Notes

- `window_size = 2`: the unbiased design forbids reusing a context, so a larger
  window reduces skip loss (see appendix).
- Absolute PPL under 4-bit quantization is depressed by repetition; treat it
  relatively.
- `no_watermark` runs reuse the same keys/params to measure the false-positive
  rate.

## Citation

```bibtex
@inproceedings{weavemark,
  title  = {Robust and Scalable Multi-bit LLM Watermarking via Coded Payload Spreading},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgements & license

`attacks/dipper.py` is adapted from
[martiansideofthemoon/ai-detection-paraphrases](https://github.com/martiansideofthemoon/ai-detection-paraphrases)
(its license applies to that file). Released under MIT (see `LICENSE`); set the
copyright holder before publishing.
