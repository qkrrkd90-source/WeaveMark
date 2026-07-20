"""WeaveMark: multi-bit LLM watermarking via coded payload spreading.

data-loading helpers live in weavemark.data (imported separately to keep the
core free of dataset deps).
"""
from weavemark.watermark import WeaveMark
from weavemark.extraction import extract_bits, detect_zerobit
from weavemark.prf import prf

__all__ = ["WeaveMark", "extract_bits", "detect_zerobit", "prf"]
