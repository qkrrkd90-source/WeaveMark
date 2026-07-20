"""DIPPER paraphraser wrapper.

Adapted from https://github.com/martiansideofthemoon/ai-detection-paraphrases
(fixed the paraphrase_batch chunk-counting bug; added single-text paraphrase
and a 4-bit option).
"""
import time

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl",
                 quantize_4bit=True, verbose=True):
        time1 = time.time()

        # run via attacks/; weavemark not importable. 4-bit pins cuda:0 and has
        # no CPU backend; --no_quantize is for larger-VRAM GPUs, not CPU.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DIPPER requires a CUDA GPU (T5-XXL, ~11B parameters). "
                "torch.cuda.is_available() returned False -- check `nvidia-smi` "
                "and that your torch build is a CUDA one "
                "(torch.__version__ should end in '+cuXXX')."
            )
        props = torch.cuda.get_device_properties(0)
        if verbose:
            print(f"[device] cuda:0 | {props.name} | {props.total_memory / 1024 ** 3:.1f} GiB")

        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')

        if quantize_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                model,
                quantization_config=quant_config,
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                model, torch_dtype=torch.bfloat16
            )
            self.model.cuda()

        if verbose:
            print(f"[DIPPER] {model} loaded ({time.time() - time1:.1f}s, 4bit={quantize_4bit})")

        self.model.eval()
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda:0")

    def _build_chunks(self, input_text, prefix, lex_code, order_code, sent_interval):
        """Split text into sent_interval-sentence DIPPER chunks."""
        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())

        chunks = []
        for sent_idx in range(0, len(sentences), sent_interval):
            curr = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            s = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                s += f" {prefix}"
            s += f" <sent> {curr} </sent>"
            chunks.append(s)
        return chunks

    def paraphrase(self, input_text, lex_diversity, order_diversity,
                   prefix="", sent_interval=3, **kwargs):
        """Paraphrase a single text."""
        assert lex_diversity in [0, 20, 40, 60, 80, 100]
        assert order_diversity in [0, 20, 40, 60, 80, 100]

        lex_code = 100 - lex_diversity
        order_code = 100 - order_diversity

        chunks = self._build_chunks(input_text, prefix, lex_code, order_code, sent_interval)
        if not chunks:
            return ""

        tok = self.tokenizer(chunks, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}

        with torch.inference_mode():
            outputs = self.model.generate(**tok, **kwargs)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return " ".join(decoded)

    def paraphrase_batch(self, input_texts, lex_diversity, order_diversity,
                         prefixes=None, sent_interval=3, **kwargs):
        """Paraphrase many texts (tracks chunk owners)."""
        assert lex_diversity in [0, 20, 40, 60, 80, 100]
        assert order_diversity in [0, 20, 40, 60, 80, 100]

        if isinstance(prefixes, str):
            prefixes = [prefixes] * len(input_texts)
        elif prefixes is None:
            prefixes = [""] * len(input_texts)

        lex_code = 100 - lex_diversity
        order_code = 100 - order_diversity

        all_chunks = []
        chunk_owner = []  # which input_text each chunk belongs to

        for i, (text, prefix) in enumerate(zip(input_texts, prefixes)):
            chunks = self._build_chunks(text, prefix, lex_code, order_code, sent_interval)
            for c in chunks:
                all_chunks.append(c)
                chunk_owner.append(i)

        if not all_chunks:
            return [""] * len(input_texts)

        tok = self.tokenizer(all_chunks, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}

        with torch.inference_mode():
            outputs = self.model.generate(**tok, **kwargs)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        per_text = [[] for _ in input_texts]
        for owner_idx, dec in zip(chunk_owner, decoded):
            per_text[owner_idx].append(dec)

        return [" ".join(parts) for parts in per_text]


if __name__ == "__main__":
    dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")

    prompt = "Tracy is a fox."
    input_text = "It is quick and brown. It jumps over the lazy dog."

    out = dp.paraphrase(input_text, lex_diversity=60, order_diversity=60,
                        prefix=prompt, do_sample=False, max_length=512)
    print(f"Input  = {prompt} <sent> {input_text} </sent>")
    print(f"Output (L60-O60 greedy) = {out}\n")
