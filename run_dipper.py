"""DIPPER attack pipeline: crop-then-paraphrase, then stride=999 whole-text
detection. Needs a run folder from generate.py / run_main.py."""
import os
import subprocess

# Models
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
PPL_MODEL_NAME = "google/gemma-2-9b"
DIPPER_MODEL = "kalpeshk2011/dipper-paraphraser-xxl"  # T5-XXL ~11B

# Modules
RUN_DIPPER = True
RUN_DETECTION = True

# (lexical, order) diversity: (60,0) lexical only; (60,60) lexical+order
DIPPER_CONFIGS = [(20, 0), (0, 20), (20, 20)]

# DIPPER inference options
DIPPER_BATCH_SIZE = 3   # reduce to 2 or 1 on OOM
SENT_INTERVAL = 3       # sentences per paraphrase chunk (DIPPER default)
MAX_LENGTH = 320        # T5 generate max_length
NO_QUANTIZE = False     # True -> full bf16 load (needs 24GB+ VRAM)

# Detection (stride=999 -> per-crop whole-text evaluation)
FORCE_WHOLE_STRIDE = 999
ZEROBIT_THRESHOLD = 4.0  # z-score threshold (None disables z_success)

# Target folder ("latest" or a specific path).
EVAL_TARGET_DIR = "latest"


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


def main():
    print("[WeaveMark] DIPPER paraphrase pipeline")

    target_dir = get_latest_output_dir() if EVAL_TARGET_DIR.lower() == "latest" else EVAL_TARGET_DIR
    if not target_dir or not os.path.exists(target_dir):
        print(f"[error] Could not find folder: {target_dir}")
        return

    print(f"target_dir = {target_dir}")
    print(f"DIPPER configs = {DIPPER_CONFIGS}")

    files_to_eval = []

    # paraphrase (one run per setting)
    for lex, order in DIPPER_CONFIGS:
        out_file = f"dipper_text_L{lex}_O{order}.jsonl"
        files_to_eval.append(out_file)

        if RUN_DIPPER:
            print(f"\n[Step 1] DIPPER L={lex}, O={order}")
            cmd_dipper = [
                "python", os.path.join("attacks", "dipper_attack.py"),
                "--data_dir", target_dir,
                "--input_file", "generation_text.jsonl",
                "--model_name", MODEL_NAME,
                "--dipper_model", DIPPER_MODEL,
                "--lex_diversity", str(lex),
                "--order_diversity", str(order),
                "--batch_size", str(DIPPER_BATCH_SIZE),
                "--sent_interval", str(SENT_INTERVAL),
                "--max_length", str(MAX_LENGTH),
            ]
            if NO_QUANTIZE:
                cmd_dipper.append("--no_quantize")
            subprocess.run(cmd_dipper, check=True)

    if not RUN_DIPPER:
        print("\n[Step 1] DIPPER skipped (reusing existing paraphrase results)")

    # detect (whole-text)
    if RUN_DETECTION:
        print(f"\n[Step 2] Running extractor (whole-text, stride={FORCE_WHOLE_STRIDE})")
        for text_file in files_to_eval:
            print(f"\n  -> {text_file}")
            cmd_eval = [
                "python", "detect.py",
                "--data_dir", target_dir,
                "--text_file", text_file,
                "--ppl_model_name", PPL_MODEL_NAME,
                "--detect",
                "--stride", str(FORCE_WHOLE_STRIDE),
            ]
            if ZEROBIT_THRESHOLD is not None:
                cmd_eval.extend(["--zerobit_threshold", str(ZEROBIT_THRESHOLD)])
            subprocess.run(cmd_eval, check=True)

    print("\nDIPPER pipeline finished.")


if __name__ == "__main__":
    main()
