"""Downstream quality pipeline: encodes a random message and runs
evaluate_downstream.py for summarization and/or translation."""
import os
import subprocess
import random

import numpy as np

from weavemark.ecc import Golay24, ReedMuller32_16

# toggles
RUN_SUMMARIZATION = True
RUN_TRANSLATION = True
RUN_NO_WATERMARK = False  # also run the no-watermark baseline

# models
SUM_MODEL = "facebook/bart-large-cnn"
TRA_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
SUM_SCORER = "roberta-large"
TRA_SCORER = "xlm-roberta-large"

# core settings
NUM_SAMPLES = 3000
OUTPUT_DIR = "output_quality_evaluation"

# ECC method: "none", "golay", "rm"
ECC_METHOD = "rm"
MSG_LEN = 16  # message length when ECC_METHOD = "none"

# WeaveMark parameters
DELTA = 1.0        # watermark strength
BPT = 0            # bits per token (kappa)
TOP_K = 50
TEMP = 1.0
WINDOW_SIZE = 4
LAYER_SHUFFLE = True   # False = no-shuffle ablation
NUM_LAYERS = 0     # number of multi-bit layers
DO_SAMPLE = True   # True: sampling; False: beam search (recommended for quality reproduction)

# Zero-bit settings (same as run_main.py)
ENABLE_ZEROBIT = True
NUM_ZEROBIT_LAYERS = 10
ZEROBIT_PARTITION_KEY = 998877
ZEROBIT_C_KEY = 665544

RESCALE_BERTSCORE = True


def run_eval(task, model, codeword, no_wm=False):
    scorer = SUM_SCORER if task == "summarization" else TRA_SCORER
    cmd = [
        "python", "evaluate_downstream.py",
        "--task", task,
        "--model_name", model,
        "--scorer_model", scorer,
        "--num_samples", str(NUM_SAMPLES),
        "--output_dir", OUTPUT_DIR,
        "--top_k", str(TOP_K),
        "--temperature", str(TEMP),
        "--window_size", str(WINDOW_SIZE)
    ]

    if not LAYER_SHUFFLE:
        cmd.append("--no_layer_shuffle")
    if DO_SAMPLE:
        cmd.append("--do_sample")
    if RESCALE_BERTSCORE:
        cmd.append("--use_rescale")
    if no_wm:
        cmd.append("--no_watermark")
    else:
        cmd.extend([
            "--bpt", str(BPT),
            "--num_layers", str(NUM_LAYERS),
            "--delta", str(DELTA),
            "--bits", codeword,
            "--num_zerobit_layers", str(NUM_ZEROBIT_LAYERS),
            "--zerobit_partition_key", str(ZEROBIT_PARTITION_KEY),
            "--zerobit_c_key", str(ZEROBIT_C_KEY)
        ])
        if ENABLE_ZEROBIT:
            cmd.append("--enable_zerobit")

    print(f"\n[{'Baseline' if no_wm else 'Watermark'}] {task.upper()} starting...")
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # encode per ECC method
    ecc_type = ECC_METHOD.lower()
    if ecc_type == 'golay':
        msg_bits = np.random.randint(0, 2, 12).tolist()
        encoded_bits = Golay24().encode(msg_bits)
    elif ecc_type == 'rm':
        msg_bits = np.random.randint(0, 2, 16).tolist()
        encoded_bits = ReedMuller32_16().encode(msg_bits)
    else:  # "none"
        msg_bits = [random.choice([0, 1]) for _ in range(MSG_LEN)]
        encoded_bits = msg_bits

    codeword = "".join(map(str, encoded_bits))
    orig_msg_str = "".join(map(str, msg_bits))

    print(f"\n[downstream] ecc={ECC_METHOD.upper()} "
          f"msg={orig_msg_str} ({len(orig_msg_str)}b) -> "
          f"codeword {len(codeword)}b, num_layers={NUM_LAYERS}")

    if RUN_SUMMARIZATION:
        if RUN_NO_WATERMARK:
            run_eval("summarization", SUM_MODEL, codeword, no_wm=True)
        run_eval("summarization", SUM_MODEL, codeword, no_wm=False)

    if RUN_TRANSLATION:
        if RUN_NO_WATERMARK:
            run_eval("translation", TRA_MODEL, codeword, no_wm=True)
        run_eval("translation", TRA_MODEL, codeword, no_wm=False)

    print("\nAll evaluations finished.")


if __name__ == "__main__":
    main()