"""Perplexity scorer: 4-bit causal LM, scores the continuation only."""
import logging
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LocalModel:

    def __init__(self, model_name: str, device=None):
        logging.info(f'Loading model from: `{model_name}`')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            ),
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

    def get_perplexity(self, prompt: str, input_texts: List[str], *args, **kwargs):
        """Perplexity of each continuation, masking the prompt.

        Args:
            prompt: Prompt prepended to each input text.
            input_texts: Continuations to score.

        Returns:
            A list of perplexity values (``-1`` on failure).
        """
        ppl_list = []

        # prompt length in tokens
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(self.model.device)
        prompt_len = prompt_ids.size(1)

        for text in input_texts:
            # empty -> -1
            if not text or len(text.strip()) == 0:
                ppl_list.append(-1)
                continue

            full_text = prompt + text
            inputs = self.tokenizer(full_text, return_tensors='pt').to(self.model.device)
            input_ids = inputs.input_ids

            # prompt >= full sequence: cannot score
            if prompt_len >= input_ids.size(1):
                ppl_list.append(-1)
                continue

            # mask prompt span with -100
            target_ids = input_ids.clone()
            target_ids[:, :prompt_len] = -100

            # labels -> mean NLL over unmasked tokens
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

            if torch.isnan(outputs.loss):
                ppl_list.append(-1)
            else:
                ppl = torch.exp(outputs.loss).item()
                ppl_list.append(ppl)

        return ppl_list
