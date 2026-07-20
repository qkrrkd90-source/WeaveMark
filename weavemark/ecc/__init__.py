"""ECC codes and soft-decision decoders: Golay[24,12,8], RM[32,16,8], and
dual-block variants for 24/32-bit messages."""
from weavemark.ecc.golay import Golay24, DualGolay48
from weavemark.ecc.reed_muller import ReedMuller32_16, DualRM64
from weavemark.ecc.decoders import (
    build_golay_soft_decoder,
    build_dual_golay_soft_decoder,
    build_rm_soft_decoder,
    build_dual_rm_soft_decoder,
)

# name -> (factory, msg_len, codeword_len)
ECC_REGISTRY = {
    "golay": (Golay24, 12, 24),
    "dual_golay": (DualGolay48, 24, 48),
    "rm": (ReedMuller32_16, 16, 32),
    "dual_rm": (DualRM64, 32, 64),
}

__all__ = [
    "Golay24",
    "DualGolay48",
    "ReedMuller32_16",
    "DualRM64",
    "build_golay_soft_decoder",
    "build_dual_golay_soft_decoder",
    "build_rm_soft_decoder",
    "build_dual_rm_soft_decoder",
    "ECC_REGISTRY",
]
