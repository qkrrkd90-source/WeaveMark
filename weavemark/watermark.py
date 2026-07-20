"""WeaveMark logits processor (embedding).

Multi-bit-per-token (kappa bits/token over ell layers, context-shuffled layer->
bit assignment) plus optional context-partitioned zero-bit layers. Each
reweighting direction is XORed with a context coin flip to stay unbiased.
"""
import time

import numpy as np
import torch
from transformers import LogitsProcessor

from weavemark.prf import prf


class WeaveMark(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        device,
        vocab_size: int,
        top_k: int = 50,
        partition_seeds: list = list(range(10)),
        c_key: int = 530773,
        bit_idx_key: int = 283519,
        shuffle_key: int = 12345,
        layer_shuffle: bool = True,   # False reproduces the no-shuffle ablation
        delta: float = 1.0,
        window_size: int = 2,  # context window h (paper uses h = 2)
        bits: str = "0",
        alpha: int = 1,
        bits_per_token: int = 1,
        # zero-bit params
        enable_zerobit: bool = True,
        num_zerobit_layers: int = 1,        # number of zero-bit layers
        zerobit_partition_key: int = 998877,  # base partition key (+i per layer)
        zerobit_c_key: int = 665544,        # base c key (+i per layer)
    ):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.device = device
        self.top_k = top_k

        # multi-bit partitions
        self.partition_masks = []
        if type(partition_seeds) is not list:
            partition_seeds = [partition_seeds]
        for key in partition_seeds:
            num_V0 = int(self.vocab_size * 0.5)
            rng = np.random.default_rng(key)
            mask = np.zeros(self.vocab_size, dtype=bool)
            mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
            mask = torch.tensor(mask).to(torch.bool).to(device)
            self.partition_masks.append(mask)

        self.c_key = c_key
        self.bit_idx_key = bit_idx_key
        self.shuffle_key = shuffle_key
        self.layer_shuffle = layer_shuffle

        self.delta = delta
        self.alpha = alpha
        self.window_size = window_size
        self.cnt = 0
        self.bits = bits
        self.hist = []

        self.bits_per_token = bits_per_token

        # zero-bit config
        self.enable_zerobit = enable_zerobit
        self.num_zerobit_layers = num_zerobit_layers if enable_zerobit else 0
        self.zerobit_partition_key = zerobit_partition_key
        self.zerobit_c_key = zerobit_c_key

        num_multibit_layers = len(self.partition_masks)
        self.total_layers = self.num_zerobit_layers + num_multibit_layers

        self._build_layer_assignment()

    def _build_layer_assignment(self):
        """Layer-type assignment; interleaves zero-bit among multi-bit when shuffled.

        layer_shuffle=False: [Z, Z, Z, M, M, M, M, M, M, M]  (zero-bit first)
        layer_shuffle=True:  [Z, M, M, Z, M, Z, M, M, M, M]  (interleaved)

        This is the same switch and key that spread the multi-bit layer->bit
        assignment, so "layer shuffling" is one setting across the scheme.

        ``layer_assignment[i] = (type, type_index)`` where
        ``type`` is 0 for zero-bit and 1 for multi-bit, and ``type_index`` is
        the index within that type.
        """
        num_multibit = len(self.partition_masks)

        layer_assignment = []
        for i in range(self.num_zerobit_layers):
            layer_assignment.append((0, i))  # zero-bit
        for i in range(num_multibit):
            layer_assignment.append((1, i))  # multi-bit

        if self.layer_shuffle and self.num_zerobit_layers > 0:
            rng = np.random.default_rng(self.shuffle_key)
            rng.shuffle(layer_assignment)

        self.layer_assignment = layer_assignment
        # zero-bit layer indices (used at detection)
        self.zerobit_layer_indices = [i for i, (t, _) in enumerate(layer_assignment) if t == 0]

    def _generate_zerobit_mask(self, seed):
        num_V0 = int(self.vocab_size * 0.5)
        rng = np.random.default_rng(seed)
        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[rng.choice(self.vocab_size, num_V0, replace=False)] = True
        return torch.tensor(mask).to(torch.bool).to(self.device)

    def _apply_zerobit_layer(self, zb_idx, prefix, prob_topk_values,
                             prob_topk_indices, skip_pos):
        """Apply one zero-bit layer (per batch element).

        The partition is message-independent: it comes only from the context and
        this layer's keys, which is what lets the detector recompute it without
        knowing the payload.
        """
        # distinct key per zero-bit layer
        zb_partition_key = self.zerobit_partition_key + zb_idx
        zb_c_key = self.zerobit_c_key + zb_idx

        zerobit_partition_seed = prf(prefix, zb_partition_key)
        zerobit_c_seed = prf(prefix, zb_c_key)

        for i in range(prefix.size(0)):
            if i in skip_pos:
                continue

            zerobit_mask = self._generate_zerobit_mask(zerobit_partition_seed[i])
            top_k_mask = zerobit_mask[prob_topk_indices[i]]

            rng_zb_c = np.random.default_rng(zerobit_c_seed[i])
            c_zerobit = rng_zb_c.integers(0, 2)
            # c=0 -> +1, c=1 -> -1
            op = 1 if c_zerobit == 0 else -1

            p0 = torch.sum(prob_topk_values[i] * top_k_mask)
            mask_p0 = (p0 < 1e-30) or (1 - p0 < 1e-30)

            if mask_p0:
                continue

            delta_val = max(min(self.alpha / p0.item(), 1 + self.delta), 1.0) - 1
            beta_val = min(delta_val * p0.item() / (1 - p0.item()), 1.0)

            delta_val = delta_val * op
            beta_val = beta_val * op

            prob_topk_values[i, top_k_mask] = prob_topk_values[i, top_k_mask] * (1 + delta_val)
            prob_topk_values[i, ~top_k_mask] = prob_topk_values[i, ~top_k_mask] * (1 - beta_val)

    def _apply_multibit_layer(self, layer_idx, prob_topk_values, prob_topk_indices,
                              ops_stack, alpha, prob_delta, skip_pos):
        top_k_mask = self.partition_masks[layer_idx][prob_topk_indices]
        p0 = torch.sum(prob_topk_values * top_k_mask, -1, keepdim=True)
        mask_p0 = (p0 < 1e-30) + (1 - p0 < 1e-30)

        delta = torch.max(torch.min(alpha / p0, 1 + prob_delta), torch.ones(prob_delta.shape).to(self.device)) - 1
        beta = torch.min(delta * p0 / (1 - p0), torch.ones(prob_delta.shape).to(self.device))

        delta[mask_p0 == 1] = 0
        beta[mask_p0 == 1] = 0

        delta[skip_pos] = 0
        beta[skip_pos] = 0

        delta = delta * ops_stack[:, layer_idx].unsqueeze(1)
        beta = beta * ops_stack[:, layer_idx].unsqueeze(1)

        delta = delta.expand(-1, prob_topk_values.shape[1])
        beta = beta.expand(-1, prob_topk_values.shape[1])

        prob_topk_values[top_k_mask == True] = prob_topk_values[top_k_mask == True] * (1 + delta)[top_k_mask == True]
        prob_topk_values[top_k_mask == False] = prob_topk_values[top_k_mask == False] * (1 - beta)[top_k_mask == False]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        begin = time.time()

        if self.cnt == 0:
            self.hist = [set() for _ in range(input_ids.shape[0])]

        if self.cnt < self.window_size:
            self.cnt += 1
            return scores

        # top-k prob distribution
        score_topk = torch.topk(scores, self.top_k, dim=-1)
        prob_topk_values = torch.nn.functional.softmax(score_topk.values, dim=-1).to(self.device).to(torch.float32)
        prob_topk_indices = score_topk.indices

        self.cnt += 1

        prob_delta = torch.full((input_ids.shape[0], 1), self.delta, dtype=torch.float32).to(self.device)
        alpha = torch.full((input_ids.shape[0], 1), self.alpha, dtype=torch.float32).to(self.device)

        inputs = input_ids.to(self.device)
        prefix = inputs[:, -self.window_size:]

        # multi-bit seeds
        c_seed = prf(prefix, self.c_key)
        bit_idx_seed = prf(prefix, self.bit_idx_key)
        shuffle_seed = prf(prefix, self.shuffle_key)

        skip_pos = []

        # skip repeated contexts (keeps reweighting unbiased)
        for i in range(prefix.size(0)):
            prefix_tuple = tuple(prefix[i].tolist())
            if prefix_tuple in self.hist[i]:
                skip_pos.append(i)
            else:
                self.hist[i].add(prefix_tuple)

        # multi-bit push directions: depend on seeds+codeword, not order, so
        # precompute before applying
        codeword_len = len(self.bits)
        num_multibit = len(self.partition_masks)
        k = self.bits_per_token
        ops_stack = None

        if num_multibit > 0:
            q = num_multibit // k
            r = num_multibit % k

            ops_stack = []

            for i in range(prefix.size(0)):
                rng_c = np.random.default_rng(c_seed[i])
                c_list = rng_c.integers(0, 2, size=num_multibit)

                rng_bit_idx = np.random.default_rng(bit_idx_seed[i])
                all_indices = np.arange(codeword_len)
                rng_bit_idx.shuffle(all_indices)
                chosen_bits_idx = all_indices[:k]

                # spread layer->bit; without it layer j always carries bit j
                layer_indices = np.arange(num_multibit)
                if self.layer_shuffle:
                    rng_shuffle = np.random.default_rng(shuffle_seed[i])
                    rng_shuffle.shuffle(layer_indices)

                layer_to_bit_map = np.zeros(num_multibit, dtype=int)
                current_layer_ptr = 0

                for j in range(k):
                    n_layers = q + 1 if j < r else q
                    for _ in range(n_layers):
                        layer_id = layer_indices[current_layer_ptr]
                        layer_to_bit_map[layer_id] = chosen_bits_idx[j]
                        current_layer_ptr += 1

                ops_list = []
                for layer_idx, c in enumerate(c_list):
                    bit_idx = layer_to_bit_map[layer_idx]
                    bit = int(self.bits[bit_idx])

                    if (c == 1 and bit == 0) or (c == 0 and bit == 1):
                        ops_list.append(1)
                    elif (c == 0 and bit == 0) or (c == 1 and bit == 1):
                        ops_list.append(-1)
                ops_stack.append(ops_list)

            ops_stack = torch.tensor(ops_stack).to(self.device)

        # apply layers in assignment order (not commutative: each delta uses the
        # current p0)
        for layer_type, type_idx in self.layer_assignment:
            if layer_type == 0:
                self._apply_zerobit_layer(
                    type_idx, prefix, prob_topk_values, prob_topk_indices, skip_pos)
            else:
                self._apply_multibit_layer(
                    type_idx, prob_topk_values, prob_topk_indices,
                    ops_stack, alpha, prob_delta, skip_pos)

        prob = torch.zeros_like(scores, dtype=torch.float32).to(self.device)
        prob.scatter_(1, prob_topk_indices, prob_topk_values)

        new_scores = torch.log(prob).to(self.device)

        return new_scores