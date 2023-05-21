import io
import zlib
from typing import IO, Any, Generic, Iterator, Optional, TypeVar

T = TypeVar("T", bound=Any)


class ReadIO(io.IOBase, Generic[T]):
    def __init__(
        self,
        stream: IO,
        chunk_size: int = io.DEFAULT_BUFFER_SIZE,
    ):
        self.stream = stream
        self.ready_buffer = bytearray()
        self.chunk_size = chunk_size
        self._closed = False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def read(self, __size: Optional[int] = None) -> T:
        raise NotImplementedError

    def readline(self, __size: Optional[int] = None) -> T:
        raise NotImplementedError

    def _process_data(self, data: bytes) -> bytes:
        return data

    def __next__(self) -> T:
        try:
            return super().__next__()  # type: ignore
        except StopIteration as stop:
            self._closed = True
            raise stop

    def __iter__(self) -> Iterator[T]:
        return self

    def _readline(self, size: int = -1) -> bytes:
        while size < 0 or len(self.ready_buffer) < size:
            if b"\n" in self.ready_buffer:
                break

            read_data = self.stream.read(self.chunk_size)
            if not read_data:
                break

            processed_data = self._process_data(read_data)
            if read_data and not processed_data:
                raise RuntimeError(f"{self.__class__.__name__} failed")

            self.ready_buffer.extend(processed_data)

        loc = self.ready_buffer.find(b"\n")
        if loc >= 0:
            return_value = self.ready_buffer[: loc + 1]
            self.ready_buffer = self.ready_buffer[loc + 1 :]
        else:
            return_value = self.ready_buffer
            self.ready_buffer = bytearray()

        if not (return_value or self.ready_buffer) and size != 0:
            # user has requested more than 0 bytes but there is nothing
            # left in the buffer to read
            raise StopIteration()

        return bytes(return_value)

    def _read(self, size: int = -1) -> bytes:
        while size < 0 or len(self.ready_buffer) < size:
            read_data = self.stream.read(self.chunk_size)
            if not read_data:
                break

            processed_data = self._process_data(read_data)
            if read_data and not processed_data:
                raise RuntimeError(f"{self.__class__.__name__} failed")

            self.ready_buffer.extend(processed_data)

        # If size equals -1, return all available data
        if size < 0:
            return_value = self.ready_buffer
            self.ready_buffer = bytearray()
        else:
            return_value = self.ready_buffer[:size]
            self.ready_buffer = self.ready_buffer[size:]

        if not (return_value or self.ready_buffer) and size != 0:
            # user has requested more than 0 bytes but there is nothing
            # left in the buffer to read
            raise StopIteration()

        return bytes(return_value)


class ReadBytesIO(ReadIO[bytes], io.RawIOBase):
    def read(self, __size: Optional[int] = None) -> bytes:
        return self._read(__size or -1)

    def readline(self, __size: Optional[int] = None) -> bytes:
        return self._readline(__size or -1)


class ReadTextIO(ReadIO[str], io.TextIOBase):
    def __init__(
        self,
        stream: IO,
        chunk_size: int = io.DEFAULT_BUFFER_SIZE,
        encoding: str = "utf-8",
        errors: str = "strict",
    ):
        super().__init__(stream=stream, chunk_size=chunk_size)
        self._encoding = encoding
        self._errors = errors

    def read(self, __size: Optional[int] = None) -> str:
        out = self._read(__size or -1)
        return out.decode(encoding=self._encoding, errors=self._errors)

    def readline(self, __size: Optional[int] = None) -> str:  # type: ignore
        out = self._readline(__size or -1)
        return out.decode(encoding=self._encoding, errors=self._errors)


class BaseZlibDecompressorIO(ReadIO[T], Generic[T]):
    def __init__(
        self,
        stream: IO,
        chunk_size: int = io.DEFAULT_BUFFER_SIZE,
        gzip: bool = True,
    ):
        gzip_offset = 16 if gzip else 0
        self.decoder = zlib.decompressobj(gzip_offset + zlib.MAX_WBITS)
        super().__init__(stream=stream, chunk_size=chunk_size)

    def _process_data(self, data: bytes) -> bytes:
        return self.decoder.decompress(data)


class BytesZLibDecompressorIO(BaseZlibDecompressorIO[bytes], ReadBytesIO):
    """Wraps a zlib decompressor so that it can be used as a file-like
    object. Returns bytes."""

    ...


class TextZLibDecompressorIO(BaseZlibDecompressorIO[str], ReadTextIO):
    def __init__(
        self,
        stream: IO,
        chunk_size: int = io.DEFAULT_BUFFER_SIZE,
        gzip: bool = True,
        encoding: str = "utf-8",
        errors: str = "strict",
    ):
        BaseZlibDecompressorIO.__init__(
            self, stream=stream, chunk_size=chunk_size, gzip=gzip
        )
        ReadTextIO.__init__(
            self,
            stream=stream,
            chunk_size=chunk_size,
            encoding=encoding,
            errors=errors,
        )

        ...
