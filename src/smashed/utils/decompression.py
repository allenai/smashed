

import zlib
from typing import IO, Optional
import io


class BaseZlibDecompressorIO:
    def __init__(
        self,
        buffered_reader: IO,
        chunk_size: int = io.DEFAULT_BUFFER_SIZE,
        is_gzip: bool = False,
    ):
        self.buffered_reader = buffered_reader

        gzip_offset = 16 if is_gzip else 0
        self.decoder = zlib.decompressobj(gzip_offset + zlib.MAX_WBITS)

        self.ready_buffer = bytearray()
        self.chunk_size = chunk_size

        self._closed = False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._closed or self.buffered_reader.closed

    def _readline(self, size: int = -1) -> bytes:
        while size < 0 or len(self.ready_buffer) < size:
            if b"\n" in self.ready_buffer:
                break

            read_data = self.buffered_reader.read(self.chunk_size)
            if not read_data:
                raise StopIteration()

            decompressed_data = self.decoder.decompress(read_data)
            self.ready_buffer.extend(decompressed_data)

        loc = self.ready_buffer.find(b"\n")
        if loc >= 0:
            return_value = self.ready_buffer[: loc + 1]
            self.ready_buffer = self.ready_buffer[loc + 1 :]
        else:
            return_value = self.ready_buffer
            self.ready_buffer = bytearray()

        return bytes(return_value)

    def _read(self, size: int = -1) -> bytes:
        while size < 0 or len(self.ready_buffer) < size:
            read_data = self.buffered_reader.read(self.chunk_size)
            if not read_data:
                raise StopIteration()

            decompressed_data = self.decoder.decompress(read_data)
            self.ready_buffer.extend(decompressed_data)

        # If size equals -1, return all available data
        if size < 0:
            return_value = self.ready_buffer
            self.ready_buffer = bytearray()
        else:
            return_value = self.ready_buffer[:size]
            self.ready_buffer = self.ready_buffer[size:]

        return bytes(return_value)


class BytesZLibDecompressorIO(io.RawIOBase, BaseZlibDecompressorIO):
    """Wraps a zlib decompressor so that it can be used as a file-like
    object. Returns bytes."""

    def __init__(
        self,
        buffered_reader: IO,
        chunk_size: int = io.DEFAULT_BUFFER_SIZE,
        is_gzip: bool = False,
    ):
        BaseZlibDecompressorIO.__init__(
            self,
            buffered_reader=buffered_reader,
            chunk_size=chunk_size,
            is_gzip=is_gzip
        )
        io.RawIOBase.__init__(self)

    def __next__(self) -> bytes:
        try:
            return super().__next__()
        except StopIteration as stop:
            self._closed = True
            raise stop

    def read(self, size: int = -1) -> bytes:
        return self._read(size)


class TextZLibDecompressorIO(io.TextIOBase, BaseZlibDecompressorIO):
    def __init__(
        self,
        buffered_reader: IO,
        chunk_size: int = io.DEFAULT_BUFFER_SIZE,
        is_gzip: bool = False,
        encoding: str = "utf-8",
        errors: Optional[str] = None,
    ):
        BaseZlibDecompressorIO.__init__(
            self,
            buffered_reader=buffered_reader,
            chunk_size=chunk_size,
            is_gzip=is_gzip
        )
        io.TextIOBase.__init__(self)

        self._encoding = encoding
        self._errors = errors

    def __next__(self) -> str:  # type: ignore
        try:
            return super().__next__()
        except StopIteration as stop:
            self._closed = True
            raise stop

    def read(self, __size: Optional[int] = None) -> str:
        out = self._read(__size or -1)
        return out.decode(self._encoding, errors=(self._errors or 'strict'))

    def readline(self, __size: Optional[int] = None) -> str:  # type: ignore
        out = self._readline(__size or -1)
        return out.decode(self._encoding, errors=(self._errors or 'strict'))
