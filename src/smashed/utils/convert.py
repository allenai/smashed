"""Utility function to convert a integer to a byte string and vice versa.

Author: Luca Soldaini @soldni
"""


def int_from_bytes(b: bytes) -> int:
    """Convert a byte string to an integer."""
    return int.from_bytes(b, byteorder="big")


def bytes_from_int(i: int) -> bytes:
    """Convert an integer to a byte string."""
    return i.to_bytes((i.bit_length() + 7) // 8, byteorder="big")
