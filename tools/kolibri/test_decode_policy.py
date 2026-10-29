"""Exercise the engine's actual host-only attention bucket function without a GPU."""
import ctypes
from pathlib import Path
import re
import subprocess

import pytest


@pytest.fixture(scope="module")
def bucket(tmp_path_factory):
    source = (Path(__file__).resolve().parents[2] / "src/kolibri/kolibri_engine.cu").read_text()
    match = re.search(r"static inline int attn_dp_bucket\([^\n]+\) \{\n.*?\n\}", source, re.S)
    assert match, "engine bucket function not found"
    work = tmp_path_factory.mktemp("decode-policy")
    cpp = work / "policy.cpp"
    cpp.write_text(match.group() + '\nextern "C" int bucket(int n, int cap) { return attn_dp_bucket(n, cap); }\n')
    subprocess.run(["c++", "-std=c++20", "-shared", "-fPIC", str(cpp), "-o", str(work / "policy.so")],
                   check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(work / "policy.so"))
    library.bucket.argtypes = [ctypes.c_int, ctypes.c_int]
    library.bucket.restype = ctypes.c_int
    return library.bucket


@pytest.mark.parametrize("n,cap", [(1, 256), (256, 256), (511, 11000), (8192, 11000),
                                   (8193, 11000), (11000, 11000)])
def test_bucket_covers_positions_without_exceeding_single_pass_cap(bucket, n, cap):
    assert n <= bucket(n, cap) <= cap


@pytest.mark.parametrize("n,want", [(1, 512), (512, 512), (513, 1024), (8192, 8192)])
def test_existing_in_range_bucket_sizes_unchanged(bucket, n, want):
    assert bucket(n, 11000) == want
