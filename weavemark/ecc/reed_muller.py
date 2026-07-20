"""ReedMuller32_16: RM [32,16,8] (16-bit msgs, codebook precomputed).
DualRM64: two blocks -> [64,32] for 32-bit msgs."""
import numpy as np


class ReedMuller32_16:
    def __init__(self):
        self.m = 5
        self.n = 32
        self.k = 16
        self._build_generator_matrix()
        self._build_ml_decoder_codebook()

    def _build_generator_matrix(self):
        X = np.array([list(map(int, format(i, '05b'))) for i in range(32)]).T
        rows = [np.ones(32, dtype=int)] + [X[i] for i in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                rows.append(X[i] * X[j])
        self.G = np.vstack(rows)

    def _build_ml_decoder_codebook(self):
        self.all_msgs_int = np.arange(65536)
        all_msgs = np.array([list(map(int, format(i, '016b'))) for i in range(65536)], dtype=int)
        self.all_msgs = all_msgs
        all_codewords = (all_msgs @ self.G) % 2
        self.codebook_polar = 2 * all_codewords - 1

    def encode(self, msg_bits):
        msg_array = np.array(msg_bits)
        codeword = (msg_array @ self.G) % 2
        return codeword.tolist()

    def decode_soft(self, soft_votes):
        correlations = self.codebook_polar @ np.array(soft_votes)
        best_idx = np.argmax(correlations)
        return self.all_msgs[best_idx].tolist()


class DualRM64:
    def __init__(self):
        self.rm = ReedMuller32_16()
        self.msg_len = 32
        self.code_len = 64

    def encode(self, msg_bits):
        if len(msg_bits) != 32:
            raise ValueError(f"Message must be 32 bits (got {len(msg_bits)}).")

        m1 = list(msg_bits[:16])
        m2 = list(msg_bits[16:])

        c1 = self.rm.encode(m1)
        c2 = self.rm.encode(m2)

        return c1 + c2
