"""Synonym substitution attack (crop-then-attack): crop each text to fixed token
lengths, then replace a fraction of content words with BERT-ranked WordNet
synonyms. One output file per ratio."""
import argparse
import os
import json
import random

import torch
import nltk
from nltk.corpus import wordnet, stopwords
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


class SynonymSubstitutionAttacker:
    def __init__(self, device=0, model_id="meta-llama/Meta-Llama-3-8B"):
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Loading BERT fill-mask pipeline...")
        self.unmasker = pipeline('fill-mask', model='bert-base-uncased', device=device)
        self.stop_words = set(stopwords.words('english'))
        self.detokenizer = TreebankWordDetokenizer()

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        return None

    def attack_text(self, text, ratio):
        llama_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_llama_token_count = len(llama_tokens)
        target_token_count = int(total_llama_token_count * ratio)

        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        target_indices = []
        for i, (word, tag) in enumerate(pos_tags):
            if word.lower() not in self.stop_words and word.isalpha() and len(word) > 2:
                wn_pos = self.get_wordnet_pos(tag)
                word_tokens = self.tokenizer.encode(" " + word, add_special_tokens=False)
                target_indices.append({
                    'index': i, 'word': word, 'wn_pos': wn_pos, 'weight': len(word_tokens)
                })

        if target_token_count == 0 or not target_indices:
            return text, 0.0

        random.shuffle(target_indices)

        replaced_token_count = 0
        for item in target_indices:
            if replaced_token_count >= target_token_count:
                break

            idx, original_word, wn_pos, weight = item['index'], item['word'], item['wn_pos'], item['weight']

            synonyms = set()
            if wn_pos:
                for syn in wordnet.synsets(original_word, pos=wn_pos):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != original_word.lower() and synonym.isalpha():
                            synonyms.add(synonym.lower())

            masked_words = words.copy()
            masked_words[idx] = '[MASK]'
            masked_text = " ".join(masked_words)

            try:
                predictions = self.unmasker(masked_text, top_k=100)
                best_replacement = None

                # prefer a WordNet-synonym prediction
                for pred in predictions:
                    pred_word = pred['token_str'].strip().lower()
                    if pred_word in synonyms:
                        best_replacement = pred_word
                        break

                # else top fluent replacement
                if not best_replacement:
                    for pred in predictions:
                        pred_word = pred['token_str'].strip().lower()
                        if pred_word != original_word.lower() and pred_word.isalpha() and len(pred_word) > 2:
                            best_replacement = pred_word
                            break

                if best_replacement:
                    words[idx] = best_replacement
                    replaced_token_count += weight
            except Exception:
                continue

        actual_ratio = replaced_token_count / total_llama_token_count
        attacked_text = self.detokenizer.detokenize(words)
        return attacked_text, actual_ratio


def main(args):
    # run as `python attacks/substitution.py`; weavemark not importable here
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The substitution attack requires a CUDA GPU: it runs a BERT "
            "fill-mask over every candidate word of every crop, which is not "
            "tractable on CPU. torch.cuda.is_available() returned False -- "
            "check `nvidia-smi` and that your torch build is a CUDA one "
            "(torch.__version__ should end in '+cuXXX')."
        )
    props = torch.cuda.get_device_properties(0)
    print(f"[device] cuda:0 | {props.name} | {props.total_memory / 1024 ** 3:.1f} GiB")

    attacker = SynonymSubstitutionAttacker(device=0, model_id=args.model_name)

    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # crop lengths (paper sweep)
    target_lengths = [25, 50, 75, 100, 125, 150, 175, 200]

    for ratio in args.ratios:
        output_file = f"attacked_text_{int(ratio * 100)}.jsonl"
        output_path = os.path.join(args.data_dir, output_file)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in tqdm(data, desc=f"Ratio {ratio} (crop & attack)"):
                original_text = item.get('generation_text', '')
                if not original_text.strip():
                    continue

                tokens = attacker.tokenizer.encode(original_text, add_special_tokens=False)

                for length in target_lengths:
                    if len(tokens) < length:
                        continue

                    # crop
                    cropped_tokens = tokens[:length]
                    cropped_text = attacker.tokenizer.decode(cropped_tokens, skip_special_tokens=True)

                    # attack the crop
                    attacked_text, act_ratio = attacker.attack_text(cropped_text, ratio)

                    new_item = item.copy()
                    new_item['cropped_length'] = length
                    new_item['attacked_text'] = attacked_text
                    new_item['target_ratio'] = ratio
                    new_item['actual_ratio'] = act_ratio

                    f.write(json.dumps(new_item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, default="generation_text.jsonl")
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    args = parser.parse_args()
    main(args)
