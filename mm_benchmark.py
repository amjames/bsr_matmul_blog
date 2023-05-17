import torch
import itertools
import csv
from pathlib import Path
import sys

try:
    from torch.sparse._triton_ops import bsr_dense_mm
except:
    def bsr_dense_mm(*args, **kwargs):
        raise Exception("Calling triton kernel from a pytorch version without it!")


def benchmark_torch_function(iters, f, *args, **kwargs):
    # warmup
    f(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.time()
    for i in range(iters):
        f(*args, **kwargs)
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)
    else:
        return (time.time() - t0)

def guard_oom_error(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.memory.empty_cache()
        return None, "OOM"
    except RuntimeError as e:
        return None, str(e)

def sparsity_ratio(dense, sparse):
    """Calculates the fraction of un-materialized values. For block sparse this accounts for the fact that nnz
    described the number of materialized blocks each of which has a square block-size x block-size shape"""
    nnz = sparse._nnz()
    numel = dense.numel()
    if sparse.layout == torch.sparse_bsr:
        bs = sparse.values().shape[-1]
        numel = dense.numel() / bs / bs

    return 1 - (nnz / numel)

def value_sparse_gen(m, n, p, dtype, device):
    """Generate m x n  random dense matrix with values >= p zeroed out"""
    return torch.randn(m, n, device=device).clamp(p).sub(p).to(dtype)

def block_sparse_gen(m, n, p, dtype, device, blocksize=None):
    """
    Generate mb x nb (<x>b = x/blocksize) random dense matrix with values >= p zeroed out. The result is expanded to
    The full m x n shape so that zeros are intentionally grouped into blocks.
    """
    mb = m // blocksize
    nb = n // blocksize
    block_A = torch.randn(mb, nb, device=device).clamp(p).sub(p).view(mb, nb, 1, 1)
    return (block_A * torch.randn(blocksize, blocksize, device=device)).transpose(-3, -2).reshape(m, n).to(dtype)

def torch_linear(x, W, out):
    return torch.nn.functional.linear(x, W, out=out)

def torch_bsr_linear(x, W, out):
    out_dims = out.shape
    mm_shape = x.shape[0] * x.shape[1], x.shape[2]
    return torch.mm(
            W,
            x.reshape(mm_shape).transpose(-2, -1),
            out=out.reshape(mm_shape).transpose(-2,-1)).transpose(-2,-1).reshape(*out_dims)

def bsr_triton_mm_linear(x, W, out):
    """
    Implement a linear operator, without bias using the triton sparse-dense matmul.

    The transposition of x and out reconciles two incompatibilities:
    1) bsr_dense_mm() requires bsr first argument. Linear computes xW^T.
    2) linear guarantees that the out passed row-major contiguous will return the same, so we transpose
    before passing and reverse the transpose on the result.

    This computes linear(x, W)^T = (xW^T)^T = Wx^T
        linear(x, W)^T^T = (Wx^T)^T = linear(x, W)

    Bias vector addition is omitted for simplicity
    """

    return bsr_dense_mm(W, x.transpose(-2,-1), out=out.transpose(-2, -1)).transpose(-2,-1)

def experiment_key_to_kernel_name(experiment_key, dtype):
    """
    Using the unmodified functionality, float32 for bsrmm directs to cusparse,
    16 bit fp type direct to a custom kernel we have implemented
    """
    if experiment_key == 'torch-2.0':
        if dtype == torch.float32:
            return "torch-2.0(cuSPARSE)"
        else:
            return "torch-2.0(fallback)"
    if experiment_key == 'torch-2.1':
        return "torch-2.1(triton)"

SIZES = [
        2048,
        #4096
]

# batchsize powers of 2 (1-256)
N_BATCH = [int(2**x) for x in range(9)]

DTYPES = [
        torch.bfloat16,
        torch.float16,
        torch.float32
]
RESULTS_HEADER = ["kernel", "dtype", "size", "blocksize", "n_batch", "sparsity", "speedup", 'note']
RESULTS_FILE = "RESULTS.csv"


def collect_experiments_speedup(key_func_map):
    # these are values which clamp + sub will leave ~50-99% of the data zero
    percentiles = torch.arange(start=0, end=2.33+.233, step=.233, dtype=float).tolist()
    blocksize_converter = {
            # 2: lambda m: m.to_sparse_bsr((2, 2)),
            # 4: lambda m: m.to_sparse_bsr((4, 4)),
            # 8: lambda m: m.to_sparse_bsr((8 ,8)),
            16: lambda m: m.to_sparse_bsr((16 ,16)),
            32: lambda m: m.to_sparse_bsr((32, 32)),
            64: lambda m: m.to_sparse_bsr((64, 64))
    }
    iters = 5
    with open(RESULTS_FILE, 'w') as rf:
        results_writer = csv.DictWriter(rf, fieldnames=RESULTS_HEADER)
        results_writer.writeheader()
        for dtype, size, n_batch in itertools.product(DTYPES, SIZES, N_BATCH):
            x = torch.randn(n_batch, size, size, dtype=dtype, device='cuda')
            out = torch.zeros(n_batch, size, size, dtype=dtype, device='cuda')
            W_dense = torch.randn(size, size, dtype=dtype, device='cuda')
            base_time = guard_oom_error(benchmark_torch_function, iters, torch_linear, x, W_dense, out)
            for kernel_key, kernel_func in key_func_map.items():
                    for blocksize, p, in itertools.product(blocksize_converter, percentiles):
                        convert_to_layout = blocksize_converter[blocksize]
                        # A must be regenerated with different blocking/sparsity in mind for each run
                        W = block_sparse_gen(size, size, p, dtype=dtype, device='cuda', blocksize=blocksize)
                        W_sparse = convert_to_layout(W)
                        note = ''
                        if base_time == 'OOM':
                            speedup = None
                            note = "Dense-OOM"
                        else:
                            time = guard_oom_error(benchmark_torch_function, iters, kernel_func, x, W_sparse, out)
                            if isinstance(time, tuple):
                                speedup, note = time
                            else:
                                speedup = f"{float(base_time) / time:3.4f}"

                        sparsity =  sparsity_ratio(W, W_sparse)
                        kernel_name = experiment_key_to_kernel_name(kernel_key, dtype)
                        results = {
                            'kernel': kernel_name,
                            'dtype': dtype,
                            'size': size,
                            'blocksize': blocksize,
                            'n_batch': n_batch,
                            'sparsity': sparsity,
                            'speedup': speedup,
                            'note': note
                        }
                        results_writer.writerow(results)
                        print(f"{kernel_name},{dtype},[batch = {n_batch}], {size} block: {blocksize}, sparsity: {sparsity} [[Speedup: {speedup}]]")
                        if len(note) != 0:
                            if note.endswith('OOM'):
                                print("** MEMORY SNAPSHOT DUE TO OOM **")
                                print(torch.cuda.memory_snapshot())
                            else:
                                print(note)
                    # Fragmentation problems occur so we clear the cache after running sparse experiments for a given group.
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    collect_experiments_speedup({
        'torch-2.0': torch_bsr_linear,
        'torch-2.1': bsr_triton_mm_linear,
        })
