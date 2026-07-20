"""Golay24: extended Golay [24,12,8]. DualGolay48: two blocks -> [48,24].
Both precompute the 4096-codeword codebook for ML decoding."""
import numpy as np


class Golay24:
    def __init__(self):
        # generator G = [I | P]
        self.P = np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
        ], dtype=int)

        self.I = np.eye(12, dtype=int)
        self.G = np.concatenate((self.I, self.P), axis=1)  # shape: (12, 24)

        # precompute codebook (4096)
        self.codewords = self._generate_all_codewords()

    def _generate_all_codewords(self):
        codewords = []
        for i in range(1 << 12):
            msg = np.array([int(b) for b in format(i, '012b')])
            code = np.dot(msg, self.G) % 2
            codewords.append((msg, code))
        return codewords

    def encode(self, msg_bits):
        msg_vec = np.array(msg_bits, dtype=int)
        code_vec = np.dot(msg_vec, self.G) % 2
        return code_vec.tolist()

    def decode_ml(self, received_str):
        """ML hard decoding with erasure handling.
        Characters 'x'/'X' are treated as erasures (excluded from the distance
        computation). Input is a 24-char string of '0', '1', 'x'; output is the
        recovered 12-bit message.
        """
        received_vec = []
        for char in received_str:
            if char == 'x' or char == 'X':
                received_vec.append(-1)  # erasure
            else:
                received_vec.append(int(char))

        best_codeword = None
        min_dist = float('inf')

        for i in range(4096):
            msg_bits = [(i >> shift) & 1 for shift in range(11, -1, -1)]
            candidate_codeword = self.encode(msg_bits)

            dist = 0
            for r, c in zip(received_vec, candidate_codeword):
                if r == -1:
                    continue
                if r != c:
                    dist += 1

            if dist < min_dist:
                min_dist = dist
                best_codeword = msg_bits
                if min_dist == 0:
                    return best_codeword

        return best_codeword


class DualGolay48:
    """Dual Golay(48, 24): 24-bit message -> 48-bit codeword.

    Built from two independent Golay(24, 12) blocks::

        message  [M1 (12) | M2 (12)]  = 24 bits
                      |         |
        codeword [C1 (24) | C2 (24)]  = 48 bits

    Each block has d_min = 8 and corrects t = 3 errors, so up to 6 errors can
    be corrected when they are split evenly across the two blocks.
    """

    def __init__(self):
        self.golay = Golay24()
        self.msg_len = 24
        self.code_len = 48
        # shared codebook for both blocks
        self.codewords = self._generate_all_codewords()

    def _generate_all_codewords(self):
        return self.golay.codewords  # 4096, shared per block

    def encode(self, msg_bits):
        if len(msg_bits) != 24:
            raise ValueError(f"Message must be 24 bits (got {len(msg_bits)}).")

        m1 = list(msg_bits[:12])
        m2 = list(msg_bits[12:])

        c1 = self.golay.encode(m1)
        c2 = self.golay.encode(m2)

        return c1 + c2

    def decode_ml(self, received_str):
        """Hard-decision ML decoding.
        Input is a 48-char string of '0', '1', 'x'; output is the recovered
        24-bit message.
        """
        if len(received_str) != 48:
            raise ValueError(f"Codeword must be 48 bits (got {len(received_str)}).")

        m1 = self.golay.decode_ml(received_str[:24])
        m2 = self.golay.decode_ml(received_str[24:])

        return m1 + m2
