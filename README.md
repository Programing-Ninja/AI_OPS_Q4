Everything needed to reproduce the Staging model is in this repository.
No out-of-band setup instructions are required.


```bash
git clone <repo-url>
cd <repo>
git checkout v1-model

mamba env create -f environment.yml     # or: conda env create -f environment.yml
mamba activate mnist_repro

dvc pull            # or `dvc checkout` if the cache is already local
python train.py
```


The reference run is recorded in `metrics.json`, which is committed at the
tagged commit. Compare your `test_accuracy` against the value there.

**Tolerance:** absolute difference in `test_accuracy` of 1e-4.

On identical CPU, identical package versions, and `torch.use_deterministic_algorithms(True)`
with `torch.set_num_threads(1)`, the run should match bit-for-bit. Anything
outside the tolerance means an environment difference, not run-to-run noise —
check `torch_version` and `platform` on the reference run's tags.


- Single seed (`--seed`, default 42) applied to `random`, `numpy`, and `torch`
- `torch.use_deterministic_algorithms(True)` and `cudnn.benchmark = False`
- `torch.set_num_threads(1)` — thread count changes float reduction order
- `DataLoader` seeded via explicit `generator` plus `worker_init_fn`
- CPU forced, so GPU model and driver version cannot cause drift
- `datasets.MNIST(download=False)` — data must come from DVC, never a fresh download
- The script refuses to run on a dirty working tree, so the `git_commit` tag
  always identifies the exact code that produced the run
- Every run is tagged with `git_commit` and `data_md5` (from `data.dvc`)


After rerunning, add a note to the *reference* run:

```bash
python note_run.py --run-id <reference-run-id> --your-accuracy <value>
```

The reference run id is in `metrics.json`.
