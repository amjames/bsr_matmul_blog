# Setup
- Use the nightly.env file to build a conda environment with the same pytorch version (nightly from 3-16 with cu11.7)
- Create a pytorch-dev conda environment, build from source following instructions [here](https://gist.github.com/amjames/96f2e83550e19d3f0bca0675b5f19776) using [this branch](https://github.com/amjames/pytorch/tree/bsr_spmm) instead
# Data collection
1. Environment pt_nightly
    - Activate environment, run baseline timing (cublas dense) and sparse speedup experiments for triton, and base
      functionality ("spmm_cu11.7")
    ```bash
    conda activate pt_nightly
    python mm_benchmark.py baseline
    python mm_benchmark.py triton
    python mm_benchmark.py spmm_cu11.7
    ```
2. Environment 2
    - Activate environment, run sample collection for build against cu12-dev
    ```bash
    conda activate pytorch-dev
    python mm_benchmark.py spmm_cu12-dev
    ```

3. Results 
    - Each file contains the speedups for a single experiment.
    - The baseline file contains baseline timing used to compute speedups. 

### Notes:
    - Generic API for spmm does not support BSR
    - Normal dispatch for BSR `torch.mm` will direct to `bsr<type>mm` legacy api, with a custom kernel used for half
      precision data types. This is used for the bfloat16/float16 data in the `spmm_cu11.7` configuration.
    - The development branch build from source disables the custom kernel for additional dtype support and will throw if the dtype is not supported
    cusparse.
    - Speedups which could not be calculated are N/A or None values, the "note" column indicates why, either the
      baseline throw OOM ("Dense-OOM") the sparse experiment threw OOM ("OOM") or another error occurred (Error is the
      note text)

