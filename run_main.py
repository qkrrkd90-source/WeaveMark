"""Clean-text TPR/FPR pipeline: generate then detect. Edit the settings below.
Attacks and downstream quality are in run_substitution/run_dipper/run_downstream.
"""
import os
import subprocess

# Models
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # e.g. mistralai/Mistral-7B-v0.1
PPL_MODEL_NAME = "google/gemma-2-9b"

# Generation
RUN_WATERMARK = True       # generate watermarked text (TPR)
RUN_NO_WATERMARK = False    # generate non-watermarked text (FPR)

NUM_TEST = 3000            # watermarked sample count
NO_WATERMARK_COUNT = 3000  # non-watermarked sample count

ECC_METHOD = "golay"          # none / golay / dual_golay / rm / dual_rm
DATASET = "c4"             # c4 / opengen / openwebtext
BATCH_SIZE = 50

MSG_LEN_NONE = 16          # message length when ECC_METHOD = "none"
MAX_NEW_TOKENS = 220
PROB_DELTA = 1.0
TEMPERATURE = 1.0
BPT = 10                    # bits per token (kappa)
NUM_LAYERS = 10             # number of multi-bit layers (ell)
TOP_K = 50
WINDOW_SIZE = 2
LAYER_SHUFFLE = True       # one switch for all layer shuffling:
                           #   - spreads the multi-bit layer->bit assignment
                           #   - interleaves zero-bit among multi-bit layers
                           # False = no-shuffle ablation

# Zero-bit layers
ENABLE_ZEROBIT = False
NUM_ZEROBIT_LAYERS = 0    # number of zero-bit layers
ZEROBIT_PARTITION_KEY = 998877
ZEROBIT_C_KEY = 665544
ZEROBIT_THRESHOLD = 4.0    # z-score threshold (None disables z-score output)

# Detection
RUN_DETECTION = True
STRIDE = 25
HARD_DECISION = False      # True: hard-decision voting
USE_LLR = False            # True: log-likelihood ratio; False: raw margin

# Perplexity
RUN_PERPLEXITY = True
PPL_LENGTH = 200

# Target folder
EVAL_TARGET_DIR = "latest"  # "latest" or a specific folder path


def get_latest_output_dir():
    base_dir = "output_dump"
    if not os.path.exists(base_dir):
        return None

    valid_dirs = []
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        params_path = os.path.join(full_path, "generation_params.json")
        if os.path.isdir(full_path) and os.path.exists(params_path):
            valid_dirs.append(full_path)

    if not valid_dirs:
        return None
    return max(valid_dirs, key=os.path.getctime)


def run_pipeline(is_no_watermark, num_samples):
    mode_str = "no-watermark (FPR)" if is_no_watermark else "watermark (TPR)"
    print(f"\n[run] {mode_str} pipeline, {num_samples} samples")

    # generate
    if is_no_watermark:
        print("\n[1/2] Generating NO-WATERMARK text...")
        cmd_gen = [
            "python", "generate.py",
            "--method", "no_watermark",
            "--model_name", MODEL_NAME,
            "--ecc_method", ECC_METHOD,
            "--num_test", str(num_samples),
            "--dataset", DATASET,
            "--max_new_tokens", str(MAX_NEW_TOKENS),
            "--temperature", str(TEMPERATURE),
            "--batch_size", str(BATCH_SIZE),
            "--top_k", str(TOP_K),
            "--window_size", str(WINDOW_SIZE),
            "--do_sample",
            "--bpt", str(BPT),
            "--num_layers", str(NUM_LAYERS),
            "--prob_delta", str(PROB_DELTA),
            "--num_zerobit_layers", str(NUM_ZEROBIT_LAYERS),
            "--zerobit_partition_key", str(ZEROBIT_PARTITION_KEY),
            "--zerobit_c_key", str(ZEROBIT_C_KEY)
        ]
    else:
        print("\n[1/2] Generating WATERMARK text...")
        cmd_gen = [
            "python", "generate.py",
            "--model_name", MODEL_NAME,
            "--ecc_method", ECC_METHOD,
            "--random_message",
            "--num_test", str(num_samples),
            "--dataset", DATASET,
            "--max_new_tokens", str(MAX_NEW_TOKENS),
            "--prob_delta", str(PROB_DELTA),
            "--temperature", str(TEMPERATURE),
            "--bpt", str(BPT),
            "--num_layers", str(NUM_LAYERS),
            "--batch_size", str(BATCH_SIZE),
            "--top_k", str(TOP_K),
            "--window_size", str(WINDOW_SIZE),
            "--do_sample",
            "--num_zerobit_layers", str(NUM_ZEROBIT_LAYERS),
            "--zerobit_partition_key", str(ZEROBIT_PARTITION_KEY),
            "--zerobit_c_key", str(ZEROBIT_C_KEY)
        ]

    if not LAYER_SHUFFLE:
        cmd_gen.append("--no_layer_shuffle")
    if ENABLE_ZEROBIT:
        cmd_gen.append("--enable_zerobit")
    if ECC_METHOD == "none":
        cmd_gen.extend(["--message", "0" * MSG_LEN_NONE])

    subprocess.run(cmd_gen, check=True)

    # Resolve target folder
    target_dir = get_latest_output_dir() if EVAL_TARGET_DIR.lower() == "latest" else EVAL_TARGET_DIR
    if not target_dir or not os.path.exists(target_dir):
        print(f"[error] Could not find a folder to analyze: {target_dir}")
        return None
    print(f"\n[*] Target directory: {target_dir}")

    # detect
    if RUN_DETECTION or RUN_PERPLEXITY:
        print("\n[2/2] Running detection...")
        cmd_eval = [
            "python", "detect.py",
            "--data_dir", target_dir,
            "--text_file", "generation_text.jsonl",
            "--ppl_model_name", PPL_MODEL_NAME
        ]

        if RUN_DETECTION:
            cmd_eval.extend(["--detect", "--stride", str(STRIDE)])
            if HARD_DECISION:
                cmd_eval.append("--hard_decision")
            if USE_LLR:
                cmd_eval.append("--use_llr")
            if ZEROBIT_THRESHOLD is not None:
                cmd_eval.extend(["--zerobit_threshold", str(ZEROBIT_THRESHOLD)])
        if RUN_PERPLEXITY:
            cmd_eval.extend(["--perplexity", "--ppl_length", str(PPL_LENGTH)])

        subprocess.run(cmd_eval, check=True)
    else:
        print("\n[2/2] Skipping detection step")

    return target_dir


def main():
    print("[WeaveMark] main pipeline (zero-bit + multi-bit)")

    watermark_dir = None
    no_watermark_dir = None

    if RUN_WATERMARK:
        watermark_dir = run_pipeline(is_no_watermark=False, num_samples=NUM_TEST)

    if RUN_NO_WATERMARK:
        no_watermark_dir = run_pipeline(is_no_watermark=True, num_samples=NO_WATERMARK_COUNT)

    print("\n[WeaveMark] pipeline done")

    if watermark_dir:
        print(f"  watermark    -> {watermark_dir}")
        print("    TPR: success_rate / z_success_rate in the summary CSV")

    if no_watermark_dir:
        print(f"  no_watermark -> {no_watermark_dir}")
        print("    FPR: z_score distribution in the detailed CSV")

    if watermark_dir and no_watermark_dir:
        print("  TPR @ FPR=1%: threshold = 99th pct of no_watermark z_score,")
        print("               then fraction of watermark samples above it.")


if __name__ == "__main__":
    main()