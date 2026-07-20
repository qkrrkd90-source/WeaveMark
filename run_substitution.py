"""Substitution attack pipeline: crop-then-attack, then stride=999 whole-text
detection. Needs a run folder from generate.py / run_main.py."""
import os
import subprocess

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
PPL_MODEL_NAME = "google/gemma-2-9b"

# Modules
RUN_ATTACK = True
ATTACK_RATIOS = [0.1, 0.2, 0.3]

RUN_DETECTION = True
# Large stride forces the extractor to evaluate each crop as a whole.
FORCE_WHOLE_STRIDE = 999

# Target folder ("latest" or a specific path).
EVAL_TARGET_DIR = "latest"


def get_latest_output_dir():
    base_dir = "output_dump"
    if not os.path.exists(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def main():
    print("[WeaveMark] synonym substitution pipeline (crop-then-attack)")

    target_dir = get_latest_output_dir() if EVAL_TARGET_DIR.lower() == "latest" else EVAL_TARGET_DIR
    if not target_dir or not os.path.exists(target_dir):
        print(f"[error] Could not find folder: {target_dir}")
        return

    files_to_eval = []

    # attack
    if RUN_ATTACK:
        print(f"\n[Step 1] Per-crop synonym substitution (ratios: {ATTACK_RATIOS})")
        cmd_attack = [
            "python", os.path.join("attacks", "substitution.py"),
            "--data_dir", target_dir,
            "--input_file", "generation_text.jsonl",
            "--ratios"
        ] + [str(r) for r in ATTACK_RATIOS]

        subprocess.run(cmd_attack, check=True)

    for ratio in ATTACK_RATIOS:
        files_to_eval.append(f"attacked_text_{int(ratio * 100)}.jsonl")

    # detect (whole-text)
    if RUN_DETECTION:
        print("\n[Step 2] Running extractor (whole-text evaluation)...")
        for text_file in files_to_eval:
            cmd_eval = [
                "python", "detect.py",
                "--data_dir", target_dir,
                "--text_file", text_file,
                "--ppl_model_name", PPL_MODEL_NAME,
                "--detect",
                "--stride", str(FORCE_WHOLE_STRIDE)
            ]
            subprocess.run(cmd_eval, check=True)

    print("\nSubstitution pipeline finished.")


if __name__ == "__main__":
    main()
