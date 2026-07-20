"""Soft-decision decoders. Each builder returns decode(soft_votes) -> bits;
picks the max-correlation codeword. Dual variants decode each half."""
import numpy as np


def build_golay_soft_decoder(golay_instance):
    all_msgs = []
    all_codes = []
    for msg, code in golay_instance.codewords:
        all_msgs.append(msg)
        all_codes.append(code)

    codebook_polar = 2 * np.array(all_codes) - 1
    all_msgs = np.array(all_msgs)

    def decode(soft_votes):
        correlations = codebook_polar @ np.array(soft_votes)
        best_idx = np.argmax(correlations)
        return all_msgs[best_idx].tolist()

    return decode


def build_dual_golay_soft_decoder(dual_golay_instance):
    golay = dual_golay_instance.golay

    all_msgs = []
    all_codes = []
    for msg, code in golay.codewords:
        all_msgs.append(msg)
        all_codes.append(code)

    codebook_polar = 2 * np.array(all_codes) - 1
    all_msgs = np.array(all_msgs)

    def decode(soft_votes):
        soft_votes = np.array(soft_votes)

        corr1 = codebook_polar @ soft_votes[:24]
        best_idx1 = np.argmax(corr1)
        m1 = all_msgs[best_idx1].tolist()

        corr2 = codebook_polar @ soft_votes[24:]
        best_idx2 = np.argmax(corr2)
        m2 = all_msgs[best_idx2].tolist()

        return m1 + m2

    return decode


def build_rm_soft_decoder(rm_instance):
    def decode(soft_votes):
        return rm_instance.decode_soft(soft_votes)

    return decode


def build_dual_rm_soft_decoder(dual_rm_instance):
    rm = dual_rm_instance.rm

    def decode(soft_votes):
        soft_votes = np.array(soft_votes)

        corr1 = rm.codebook_polar @ soft_votes[:32]
        best_idx1 = np.argmax(corr1)
        m1 = rm.all_msgs[best_idx1].tolist()

        corr2 = rm.codebook_polar @ soft_votes[32:]
        best_idx2 = np.argmax(corr2)
        m2 = rm.all_msgs[best_idx2].tolist()

        return m1 + m2

    return decode
