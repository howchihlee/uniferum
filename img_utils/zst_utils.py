import struct

import numpy as np
import zstandard as zstd


def save_zst16(filename, array: np.ndarray, compress_level: int = 6):
    assert array.dtype == np.int16
    assert array.ndim == 3
    assert isinstance(compress_level, int), "compress_level must be an integer"
    # Flatten data and convert to bytes
    flat_data = array.flatten()
    byte_data = flat_data.tobytes()

    # Pack shape info: 3 integers (X, Y, Z)
    shape_header = struct.pack("3I", *array.shape)

    if compress_level > 0:
        cctx = zstd.ZstdCompressor(level=compress_level)
        compressed = cctx.compress(byte_data)
        with open(filename, "wb") as f:
            f.write(b"ZST0")  # magic number for compressed
            f.write(shape_header)
            f.write(compressed)
    else:
        with open(filename, "wb") as f:
            f.write(b"RAW0")  # magic number for raw
            f.write(shape_header)
            f.write(byte_data)


def load_zst16(filename):
    with open(filename, "rb") as f:
        magic = f.read(4)
        shape = struct.unpack("3I", f.read(12))
        data = f.read()

    if magic == b"ZST0":
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(data)
        flat_array = np.frombuffer(decompressed, dtype=np.int16)
    elif magic == b"RAW0":
        flat_array = np.frombuffer(data, dtype=np.int16)
    else:
        raise ValueError("Unknown file format")

    return flat_array.reshape(shape)
