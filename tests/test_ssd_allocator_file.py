import os

from flexkv.storage.allocator import SSDAllocator


def test_ssd_file_is_allocated_and_reused(tmp_path):
    path = str(tmp_path / "flexkv_ssd_cache_0_0.bin")
    size = 8 << 20

    SSDAllocator._create_file(path, size)
    st = os.stat(path)
    assert st.st_size == size
    assert st.st_blocks * 512 >= size          # reserved, not sparse

    with open(path, "r+b") as f:
        f.seek(size // 2)
        f.write(b"kept")

    SSDAllocator._create_file(path, size)      # restart with the same size: kept as is
    with open(path, "rb") as f:
        f.seek(size // 2)
        assert f.read(4) == b"kept"

    SSDAllocator._create_file(path, 2 * size)  # resized and reserved again
    st = os.stat(path)
    assert st.st_size == 2 * size and st.st_blocks * 512 >= 2 * size
