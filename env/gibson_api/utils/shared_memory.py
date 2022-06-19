"""Provides shared memory for direct access across processes.
The API of this package is currently provisional. Refer to the
documentation for details.
"""


__all__ = [ 'SharedNumpyPool' ]


from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
import numpy as np


_O_CREX = os.O_CREAT | os.O_EXCL

# FreeBSD (and perhaps other BSDs) limit names to 14 characters.
_SHM_SAFE_NAME_LENGTH = 14


def _make_filename(directory):
    "Create a random filename for the shared memory object."
    # number of random bytes to use for name
    nbytes = _SHM_SAFE_NAME_LENGTH // 2
    assert nbytes >= 2
    name = secrets.token_hex(nbytes)
    assert len(name) <= _SHM_SAFE_NAME_LENGTH
    return os.path.join(directory, name)


class SharedMemory:
    """Creates a new shared memory block or attaches to an existing
    shared memory block.
    Every shared memory block is assigned a unique name.  This enables
    one process to create a shared memory block with a particular name
    so that a different process can attach to that same shared memory
    block using that same name.
    As a resource for sharing data across processes, shared memory blocks
    may outlive the original process that created them.  When one process
    no longer needs access to a shared memory block that might still be
    needed by other processes, the close() method should be called.
    When a shared memory block is no longer needed by any process, the
    unlink() method should be called to ensure proper cleanup."""

    # Defaults; enables close() and unlink() to run without errors.
    _name = None
    _fd = -1
    _mmap = None
    _buf = None
    _flags = os.O_RDWR
    _mode = 0o600

    def __init__(self, directory, name=None, create=False, size=0):
        if not size >= 0:
            raise ValueError("'size' must be a positive integer")
        if create:
            self._flags = _O_CREX | os.O_RDWR
            if size == 0:
                raise ValueError("'size' must be a positive number different from zero")
        if name is None and not self._flags & os.O_EXCL:
            raise ValueError("'name' can only be None if create=True")


        # Shared Memory
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

        if name is None:
            while True:
                name = _make_filename(directory)
                try:
                    self._fd = os.open(name, self._flags, mode=self._mode)
                except FileExistsError:
                    continue
                self._name = name
                break
        else:
            self._fd = os.open(name, self._flags, mode=self._mode)
            self._name = name
        try:
            if create and size:
                os.ftruncate(self._fd, size)
            stats = os.fstat(self._fd)
            size = stats.st_size
            self._mmap = mmap.mmap(self._fd, size)
        except OSError:
            self.unlink()
            raise

        self._size = size
        self._buf = memoryview(self._mmap)

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.name,
                False,
                self.size,
            ),
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r}, size={self.size})'

    @property
    def buf(self):
        "A memoryview of contents of the shared memory block."
        return self._buf

    @property
    def name(self):
        "Unique name that identifies the shared memory block."
        return self._name

    @property
    def size(self):
        "Size in bytes."
        return self._size

    def close(self):
        """Closes access to the shared memory from this instance but does
        not destroy the shared memory block."""
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def unlink(self):
        """Requests that the underlying shared memory block be destroyed.
        In order to ensure proper cleanup of resources, unlink should be
        called once (and only once) across all processes which have access
        to the shared memory block."""
        if self._name:
            try:
                os.remove(self._name)
            except:
                pass



class SharedNumpyPool:
    def __init__(self, directory, max_size=0, name=None, table={}, used=0):
        self.shm = None if name is None else SharedMemory(name=name, directory=directory)
        self.table = table
        self.used = used
        self.max_size = max_size
        self.lazy = False
        self.directory = directory

    def allocate_lazy(self):
        assert self.shm is None
        self.lazy = True
        yield None
        self.lazy = False
        assert self.shm is None
        self.max_size = (self.used//4096+1) * 4096
        self.used = 0
        self.shm = SharedMemory(create=True, size=self.max_size, directory=self.directory)
        yield None

    def allocate(self, key, shape, dtype=np.float32):
        assert self.table.get(key) is None
        size = np.dtype(dtype).itemsize
        for i in shape:
            size *= i
        self.used += size
        if not self.lazy:
            assert self.max_size >= size
            assert self.shm is not None
            self.table[key] = (shape, dtype, self.used - size, self.used, size // shape[0])
            return np.ndarray(shape, dtype=dtype, buffer=self.shm.buf[self.used - size:self.used])

    def allocate_copies(self, key, arr):
        return self.allocate(key, arr.shape, arr.dtype.str)

    def get(self, key, rg=None, reduce=False):
        shape, dtype, begin, end, csize = self.table[key]
        if rg is None:
            return np.ndarray(shape, dtype=dtype, buffer=self.shm.buf[begin:end])
        if reduce and rg[1] - rg[0] == 1:
            return np.ndarray(shape[1:], dtype=dtype, buffer=self.shm.buf[begin+rg[0]*csize:begin+rg[1]*csize])
        return np.ndarray((rg[1]-rg[0], *shape[1:]), dtype=dtype, buffer=self.shm.buf[begin+rg[0]*csize:begin+rg[1]*csize])

    def dump(self):
        return {
            'max_size': self.max_size,
            'name': self.shm.name,
            'table': self.table,
            'used': self.used,
            'directory': self.directory
        }

    def __del__(self):
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()

    
