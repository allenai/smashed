import gzip as gz
import io
from contextlib import contextmanager
from typing import IO, Iterator, Literal, Optional, cast

from .io_wrappers import BytesZLibDecompressorIO, TextZLibDecompressorIO


@contextmanager
def decompress_stream(
    stream: IO,
    mode: Literal["r", "rt", "rb"] = "rt",
    encoding: Optional[str] = "utf-8",
    errors: str = "strict",
    chunk_size: int = io.DEFAULT_BUFFER_SIZE,
    gzip: bool = True,
) -> Iterator[IO]:
    out: io.IOBase

    if mode == "rb" or mode == "r":
        out = BytesZLibDecompressorIO(
            stream=stream, chunk_size=chunk_size, gzip=gzip
        )
    elif mode == "rt":
        assert encoding is not None, "encoding must be provided for text mode"
        out = TextZLibDecompressorIO(
            stream=stream,
            chunk_size=chunk_size,
            gzip=gzip,
            encoding=encoding,
            errors=errors,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # cast to IO to satisfy mypy, then yield
    yield cast(IO, out)

    # Flush and close the stream
    out.close()


@contextmanager
def compress_stream(
    stream: IO,
    mode: Literal["w", "wt", "wb"] = "wt",
    encoding: Optional[str] = "utf-8",
    errors: str = "strict",
    gzip: bool = True,
) -> Iterator[IO]:
    assert gzip, "Only gzip compression is supported at this time"

    if mode == "wb" or mode == "w":
        out = gz.open(stream, mode=mode)
    elif mode == "wt":
        assert encoding is not None, "encoding must be provided for text mode"
        out = gz.open(stream, mode=mode, encoding=encoding, errors=errors)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # cast to IO to satisfy mypy, then yield
    yield cast(IO, out)

    # Flush and close the stream
    out.close()
