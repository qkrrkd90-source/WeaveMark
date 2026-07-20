"""Message extraction and zero-bit detection.

extract_bits: per-bit vote margins (soft or hard) fed to the ECC decoder.
detect_zerobit: green-token z-score over the zero-bit layers.
Seeding and layer->bit assignment must mirror WeaveMark.__call__.
"""
import numpy as np
import torch

from weavemark.prf import prf


def detect_zerobit(tokens, vocab_size, window_size=2,
                   zerobit_partition_key=998877,
                   zerobit_c_key=665544,
                   num_zerobit_layers=1):
    """Zero-bit detection over the zero-bit layers.

    Args:
        tokens: List of token ids.
        vocab_size: Vocabulary size.
        window_size: Context window size.
        zerobit_partition_key: Base partition key (+i per layer).
        zerobit_c_key: Base c key (+i per layer).
        num_zerobit_layers: Number of zero-bit layers.

    Returns:
        dict with keys ``green_count`` and ``total_count`` (summed over all
        layers), ``z_score``, ``p_value``, and ``per_layer`` (per-layer counts).
    """
    ids = tokens
    ids_tensor = torch.tensor(ids)

    per_layer_results = []

    for layer_idx in range(num_zerobit_layers):
        # distinct key per layer
        layer_partition_key = zerobit_partition_key + layer_idx
        layer_c_key = zerobit_c_key + layer_idx

        hist = set()
        green_count = 0
        total_count = 0

        for i in range(window_size, len(ids)):
            context_tensor = ids_tensor[i - window_size: i]
            current_token = ids[i]

            prefix_tuple = tuple(context_tensor.tolist())
            if prefix_tuple in hist:
                continue  # skip repeated context
            hist.add(prefix_tuple)

            seed_partition = prf(context_tensor, layer_partition_key)
            rng_partition = np.random.default_rng(seed_partition)
            num_V0 = int(vocab_size * 0.5)
            green_mask = np.zeros(vocab_size, dtype=bool)
            green_mask[rng_partition.choice(vocab_size, num_V0, replace=False)] = True

            seed_c = prf(context_tensor, layer_c_key)
            rng_c = np.random.default_rng(seed_c)
            c = rng_c.integers(0, 2)

            # c=0 -> mask True green, c=1 -> mask False green
            if c == 0:
                is_green = green_mask[current_token]
            else:
                is_green = not green_mask[current_token]

            if is_green:
                green_count += 1
            total_count += 1

        per_layer_results.append({
            'layer_idx': layer_idx,
            'green_count': green_count,
            'total_count': total_count
        })

    total_green = sum(r['green_count'] for r in per_layer_results)
    total_samples = sum(r['total_count'] for r in per_layer_results)

    # z-score, gamma=0.5
    if total_samples == 0:
        z_score = 0.0
        p_value = 0.5
    else:
        expected = total_samples / 2
        std = np.sqrt(total_samples) / 2
        z_score = (total_green - expected) / std if std > 0 else 0.0

        from scipy import stats
        p_value = 1 - stats.norm.cdf(z_score)

    return {
        'green_count': total_green,
        'total_count': total_samples,
        'z_score': z_score,
        'p_value': p_value,
        'per_layer': per_layer_results
    }


def extract_bits(tokens, processor, codeword_len, k, num_layers,
                 hard_decision=False, use_llr=False):
    """Per-bit votes -> soft (or hard) score vector.

    Args:
        tokens: List of token ids.
        processor: A ``WeaveMark`` instance (shares keys and partitions with
            the embedder).
        codeword_len: Codeword length n.
        k: Bits per token (kappa).
        num_layers: Number of multi-bit layers ell.
        hard_decision: If True, majority vote then map to polar +/-1.
        use_llr: If True (and not hard), use the log-likelihood ratio as the
            per-bit score instead of the raw vote margin.

    Combinations:
        hard_decision=False, use_llr=False -> raw margin (v1 - v0)  [default]
        hard_decision=False, use_llr=True  -> log-likelihood ratio
        hard_decision=True                 -> majority then polar (+/-1, 0)

    Returns:
        (detected_bits_str, decoder_input, counts) where ``decoder_input`` is
        the per-bit score vector fed to the soft-decision decoder and
        ``counts`` is the number of votes accumulated per bit.
    """
    ids = tokens
    ids_tensor = torch.tensor(ids, device=processor.device)

    # per-value counts (for LLR)
    votes_for_0 = np.zeros(codeword_len, dtype=int)
    votes_for_1 = np.zeros(codeword_len, dtype=int)
    counts = np.zeros(codeword_len, dtype=int)

    window = processor.window_size
    masks_cpu = [m.cpu().numpy() for m in processor.partition_masks]
    hist = set()

    if k == 0:
        return "", [0] * codeword_len, [0] * codeword_len

    q, r = num_layers // k, num_layers % k

    for i in range(window, len(ids)):
        context_tensor = ids_tensor[i - window: i]
        current_token = ids[i]

        prefix_tuple = tuple(context_tensor.tolist())
        if prefix_tuple in hist:
            continue
        hist.add(prefix_tuple)

        seed_bit = prf(context_tensor, processor.bit_idx_key)
        seed_c = prf(context_tensor, processor.c_key)
        seed_shuffle = prf(context_tensor, processor.shuffle_key)

        rng_bit = np.random.default_rng(seed_bit)
        all_indices = np.arange(codeword_len)
        rng_bit.shuffle(all_indices)
        chosen_bits_idx = all_indices[:k]

        # must mirror embedding
        layer_indices = np.arange(num_layers)
        if getattr(processor, "layer_shuffle", True):
            rng_shuffle = np.random.default_rng(seed_shuffle)
            rng_shuffle.shuffle(layer_indices)

        layer_to_bit_map = np.zeros(num_layers, dtype=int)
        current_layer_ptr = 0
        for j in range(k):
            n_layers = q + 1 if j < r else q
            for _ in range(n_layers):
                layer_to_bit_map[layer_indices[current_layer_ptr]] = chosen_bits_idx[j]
                current_layer_ptr += 1

        rng_c = np.random.default_rng(seed_c)
        c_list = rng_c.integers(0, 2, size=num_layers)

        for layer_idx, mask in enumerate(masks_cpu):
            target_bit_idx = layer_to_bit_map[layer_idx]
            c_val = c_list[layer_idx]
            predicted_bit = 1 - c_val if mask[current_token] else c_val

            if predicted_bit == 1:
                votes_for_1[target_bit_idx] += 1
            else:
                votes_for_0[target_bit_idx] += 1
            counts[target_bit_idx] += 1

    # per-bit decision + decoder input
    detected_bits = []
    decoder_input = []

    for b in range(codeword_len):
        v0, v1 = votes_for_0[b], votes_for_1[b]

        if counts[b] == 0:
            detected_bits.append('n')
            decoder_input.append(0)
        elif hard_decision:
            if v1 > v0:
                detected_bits.append('1')
                decoder_input.append(1)
            elif v0 > v1:
                detected_bits.append('0')
                decoder_input.append(-1)
            else:
                detected_bits.append('x')
                decoder_input.append(0)
        elif use_llr:
            llr = np.log((v1 + 0.5) / (v0 + 0.5))
            if llr > 0:
                detected_bits.append('1')
            elif llr < 0:
                detected_bits.append('0')
            else:
                detected_bits.append('x')
            decoder_input.append(llr)
        else:
            diff = v1 - v0
            if diff > 0:
                detected_bits.append('1')
            elif diff < 0:
                detected_bits.append('0')
            else:
                detected_bits.append('x')
            decoder_input.append(float(diff))

    return "".join(detected_bits), decoder_input, counts.tolist()
