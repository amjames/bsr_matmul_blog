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

def torch_mm_ab_t(a, b, c):
    return torch.mm(a, b.transpose(0,1), out=c)

def bsr_triton_mm(a, b, c):
    return bsr_dense_mm(a, b.transpose(0, 1), skip_checks=True, out=c)

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
BASELINE_HEADER = ["dtype", "size", "n_batch", "time"]
RESULTS_HEADER = ["kernel", "dtype", "size", "blocksize", "n_batch", "sparsity", "speedup", 'note']
BASELINE_FILE = "baseline_timings.csv"
RESULTS_FILE_SUFFIX = "results.csv"

def dtype_from_string(s):
    for dt in DTYPES:
        if str(dt) == s:
            return dt
    raise Exception(f"Could not find pytorch dtype for string: {s}")


def collect_experiment_speedup(experiment_key, objective_function):
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

    baseline_file = Path(BASELINE_FILE)
    assert baseline_file.is_file(), "Could not locate baseline raw timings"
    with open(baseline_file) as bf:
        baseline_reader = csv.DictReader(bf)
        results_file = f"{experiment_key}_{RESULTS_FILE_SUFFIX}"
        with open(results_file, 'w') as rf:
            results_writer = csv.DictWriter(rf, fieldnames=RESULTS_HEADER)
            results_writer.writeheader()

            for baseline in baseline_reader:
                dtype = dtype_from_string(baseline['dtype'])
                size = int(baseline['size'])
                n_batch = int(baseline['n_batch'])
                base_time = baseline['time']
                # Re-use these across blocksize/sparsity changes, they are not going to change
                B = torch.randn(n_batch*size, size, dtype=dtype, device='cuda')
                # pre-allocate output
                C = torch.zeros(size, n_batch*size, dtype=dtype, device='cuda')
                for blocksize, p in itertools.product(blocksize_converter, percentiles):
                    convert_to_layout = blocksize_converter[blocksize]
                    # A must be regenerated with different blocking/sparsity in mind for each run
                    A = block_sparse_gen(size, size, p, dtype, device='cuda', blocksize=blocksize)
                    A_sparse = convert_to_layout(A)
                    iters = 5
                    note = ''
                    if base_time == 'OOM':
                        speedup = None
                        note = "Dense-OOM"
                    else:
                        time = guard_oom_error(benchmark_torch_function, iters, objective_function, A_sparse, B, C)
                        if isinstance(time,tuple):
                            speedup, note = time
                        else:
                            speedup = f"{float(base_time) / time:3.4f}"

                    sparsity =  sparsity_ratio(A, A_sparse)
                    kernel_name = experiment_key_to_kernel_name(experiment_key, dtype)
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
                    if note.endswith("OOM"):
                        print("** MEMORY SNAPSHOT DUE TO OOM **")
                        print(torch.cuda.memory_snapshot())


                # Fragmentation problems occur so we clear the cache after running sparse experiments for a given group.
                torch.cuda.empty_cache()





def collect_baseline_raw_timings():
    with open(BASELINE_FILE, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=BASELINE_HEADER)
        writer.writeheader()

        for dtype, size, n_batch in itertools.product(DTYPES, SIZES, N_BATCH):
            A = torch.randn(size, size, dtype=dtype, device='cuda')
            B = torch.randn(n_batch*size, size, dtype=dtype, device='cuda')
            C = torch.zeros(size, size*n_batch, dtype=dtype, device='cuda')

            iters = 5
            time = guard_oom_error(benchmark_torch_function, iters, torch_mm_ab_t, A, B, C)

            if time is None:
                time = 'OOM'

            results = {
                'dtype': dtype,
                'size': size,
                'n_batch': n_batch,
                'time': time
            }
            writer.writerow(results)
            print(f"BASELINE: {dtype}, {n_batch}, {size}: [[TIME: {time}]]")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    collect_baseline_raw_timings()
    collect_experiment_speedup('torch-2.0', torch_mm_ab_t)
    collect_experiment_speedup('torch-2.1', bsr_triton_mm)
