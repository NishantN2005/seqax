# Restoring the SPIRe environment on a fresh instance

State as of 2026-09-03. Trained and measured: the 67M target (eval loss 2.534),
the vanilla draft (3.067), SPIRe **Phase A** (3.000, τ = 2.851 at L=512 T=1.0), and
SPIRe **Phase B** with target-activation substitution (3.661, τ = 2.769 — a
negative result). Phase 4 has its first end-to-end cost-model measurement.

Numbers live in [results.md](results.md); what they may and may not be compared
against lives in [fidelity-ledger.md](fidelity-ledger.md). Read ledger §4.1 before
trusting any τ difference under ~0.03, and §5.1 for why every τ measured before
2026-08-31 is invalid.

**Next on the critical path:** a compact sparse KV cache. It is the sole blocker on
timing SPIRe and MagicDec drafts, and their cost advantage is the paper's entire
argument. Today's cache is a mask, so a sparse draft still reads full `Klen` from
HBM — timing it would measure this implementation rather than the method.

---

## What the backup holds

`~/matX/spire_backup`, 8.2 GB. Not in git; regenerate with `~/matX/backup_vm.sh`.

| path | what |
|---|---|
| `runs/spire_target_1024/0000036203/` | target — **5.5 GPU-hours** |
| `runs/spire_draft_vanilla/0000010421/` | vanilla draft — ~45 min |
| `runs/spire_draft_spire/0000020841/` | **SPIRe Phase A** — the best SPIRe result, ~2.5 h |
| `runs/spire_draft_spire_fm/0000020841/` | SPIRe Phase B (rollout) — ~3 h, negative result |
| `runs/*/log.zarr` | seqax training metrics per run |
| `data/longcrawl64-flat-tokens.zarr` | train 2,592,079,872 / validation 213,528,576 tokens |
| `logs/` | all 24 logs: every loss curve, LR probe, τ sweep, benchmark, diagnostic |
| `results/bench_vanilla*.json` | ITM measurements (Finding 10) |
| `env/jax060-freeze.txt` | 87 pinned packages — exact training environment |
| `env/zarr2-freeze.txt` | 22 pinned packages — exact tools environment |
| `env/system.txt` | driver, CUDA, Ubuntu, python, `jax.devices()` at capture |
| `configs/spire_target_1024_resolved.yaml` | hydra-resolved config `tau.py` needs (generated, not in git) |
| `dataset_manifest.md5` | checksums of all 1,061 raw source chunks |

Deliberately absent: the raw `longcrawl64` zarr (only needed to rebuild flat-tokens
at a width other than 4096), intermediate checkpoints, the two *failed* Phase B
checkpoints (their evidence is the `w_memory` matrices, recorded in results.md
Finding 9), and the venvs — venvs do not relocate, which is why `pip freeze` is
captured instead.

**Checkpoints are zarr v3.** Completeness is `attributes.write_completed` inside
`<checkpoint>/zarr.json` — *not* `.zattrs`, which is the v2 location and will make
a checkpoint look incomplete when it is fine. The dataset, written by the zarr-2
tools venv, does use `.zarray`/`.zattrs`. Verify a backup with:

```bash
python3 -c "
import json,pathlib
for ck in pathlib.Path('~/matX/spire_backup/runs').expanduser().glob('*/0*'):
    a=json.loads((ck/'zarr.json').read_text())['attributes']
    print(ck.parent.name, a.get('write_completed'))"
```

---

## Bringing up a new box

Launch **GPU Base 24.04** — not Lambda Stack, whose preinstalled torch/JAX collide
with `jax[cuda12]`. Attach a persistent filesystem at creation if you want to stop
and restart later; Lambda has no stop-and-keep-disk state for on-demand instances,
and filesystems cannot be added to a running box.

```bash
sudo apt-get update -qq && sudo apt-get install -y graphviz python3-venv
git clone -b spire_reproducer https://github.com/NishantN2005/seqax.git ~/seqax
python3 -m venv ~/jax060 && ~/jax060/bin/pip install -q -U pip setuptools wheel
python3 -m venv ~/zarr2  && ~/zarr2/bin/pip install -q -U pip
```

Then push pins, the generated config, and the data:

```bash
B=~/matX/spire_backup
scp $B/env/*-freeze.txt $B/configs/* ubuntu@NEW_IP:/tmp/
ssh ubuntu@NEW_IP 'cp /tmp/spire_target_1024_resolved.yaml /tmp/requirements-gpu.txt ~/seqax/configs/
                   ~/jax060/bin/pip install -q -r /tmp/jax060-freeze.txt
                   ~/zarr2/bin/pip install -q -r /tmp/zarr2-freeze.txt
                   mkdir -p ~/data ~/runs'
```

Stream the data and checkpoints with **tar over ssh, not rsync**. macOS now ships
`openrsync` (protocol 29), which rejects `--info=progress2`; because that is a
usage error on stderr, a piped `rsync … | tail` swallows it and the script reports
success having transferred nothing. This cost a silent full-dataset skip the first
time this document was used. tar is also faster here — ~1,600 small files.

```bash
tar cf - -C $B/data longcrawl64-flat-tokens.zarr | ssh ubuntu@NEW_IP 'tar xf - -C ~/data'
tar cf - -C $B/runs . | ssh ubuntu@NEW_IP 'tar xf - -C ~/runs'
```

`tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.*'` is
expected and harmless. Verify by file count and the gates below, never by absence
of warnings. Expect 731 files under `~/data`; 109 per checkpoint directory.

Installing from `jax060-freeze.txt` rather than `requirements-gpu.txt` is what
makes the environment *exact* — the requirements file pins only direct
dependencies, the freeze pins all 87 including transitives.

`configs/dataset/longcrawl.yaml` and the run configs hardcode `/home/ubuntu/...`
paths. Fine on a Lambda box, wrong anywhere else — override with Hydra or edit.

To regenerate the resolved config instead of restoring it (`tau.py` needs it
because a plain `OmegaConf.load` does not process the `defaults:` list, leaving
`cfg.dataset` missing):

```bash
~/jax060/bin/python -m train --config-name=spire_target_1024 +model_name=spire_target_1024 --cfg job --resolve | grep -v '^#' > configs/spire_target_1024_resolved.yaml
```

---

## Verifying the restore

```bash
cd ~/seqax
~/zarr2/bin/python tools/verify_flat_tokens.py --zarr ~/data/longcrawl64-flat-tokens.zarr
PYTHONPATH=. ~/jax060/bin/python tools/gen_text.py    # must be coherent English
```

`gen_text.py` is the real gate. If it emits `France France France…`, the BOS
convention has regressed — run `tests/test_convention.py`, which exists precisely
to catch that and is one of only two tests that compare decode against the
*training* convention rather than against itself.

Then confirm the recorded numbers reproduce:

```bash
# self-draft floor: expect τ ≈ 4.98, floor 0.014-0.025 (a RANGE, see ledger §4.1)
PYTHONPATH=. ~/jax060/bin/python tau.py --config spire_target_1024_resolved \
  --model-name spire_target_1024 --mode self --k 4 --context-len 512 \
  --batch 32 --num-batches 2 --rounds 8 --temperature 0.0 --prompts dataset

# vanilla: expect τ ≈ 2.57 ;  SPIRe Phase A: expect τ ≈ 2.85
PYTHONPATH=. ~/jax060/bin/python tau.py --config spire_target_1024_resolved \
  --model-name spire_target_1024 --mode spire \
  --draft-config spire_draft_spire --draft-model-name spire_draft_spire \
  --k 4 --context-len 512 --window 64 --sink 1 --temperature 1.0 \
  --batch 32 --num-batches 4 --rounds 8 --prompts dataset
```

The CPU tests need a wrapper, because `init_seqax.set_variables()` force-sets
`JAX_PLATFORMS=cuda` whenever `nvidia-smi` succeeds, hiding the 8 forced CPU
devices the tests' `(8,1,1)` mesh needs:

```python
# /tmp/runtest.py
import os, sys
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import init_seqax
os.environ["JAX_PLATFORMS"] = "cpu"   # override BEFORE jax is imported
path = sys.argv[1]; sys.argv = [path]
exec(compile(open(path).read(), path, "exec"), {"__name__": "__main__", "__file__": path})
```

```bash
for t in tests/test_dense.py tests/test_streaming.py tests/test_spec.py tests/test_convention.py; do
  PYTHONPATH=. ~/jax060/bin/python /tmp/runtest.py $t
done
PYTHONPATH=. ~/jax060/bin/python tests/test_distill.py   # sets its own device count
```

---

## Rebuilding models from scratch

Order matters: the drafts distill from the target, and the SPIRe drafts are pruned
from it.

```bash
# 1. target — ~5.5 h
python -m train --config-name=spire_target_1024 +model_name=spire_target_1024

# 2. vanilla draft — ~45 min, independent of the target
python -m train --config-name=spire_draft_vanilla +model_name=spire_draft_vanilla

# 3. SPIRe Phase A — prune first, then train. ~2.5 h
PYTHONPATH=. python tools/prune_init.py \
  --target-config spire_target_1024 --target-name spire_target_1024 \
  --draft-config spire_draft_spire --draft-name spire_draft_spire
python -m train --config-name=spire_draft_spire +model_name=spire_draft_spire
```

`prune_init.py` writes step 0 into the draft's own run directory rather than using
`clone_from`, whose zarr log a never-trained checkpoint cannot provide.

---

## Operational notes, all learned the hard way

- Run anything long **detached** (`setsid nohup … > ~/log 2>&1 < /dev/null &`) and
  poll the log. Long-lived ssh sessions to these boxes get reset regularly; several
  died mid-run, one silently discarding a completed verification.
- **Never `pgrep -f` a pattern that appears in your own command line.** Over ssh the
  remote runs `bash -c '<your script>'`, so the pattern matches itself. This killed
  an ssh session and produced a permanent false "training still alive" reading.
  Match the interpreter path (`pgrep -f "[j]ax060/bin/python -m train"`) instead —
  and note the bracket trick alone is not enough if the pattern is also in the
  wrapper.
- **Redirecting python stdout to a file block-buffers it.** A long run will show an
  empty log until it exits. Use `python -u`.
- A failed run leaves a partial `log.zarr` that poisons the *next* run with
  `KeyError: 'timestamp'`. `rm -rf ~/runs/<name>/log.zarr` — not a real failure.
- `dataset.eval_tokens` must divide evenly by `batch_size * seq_len`, or training
  aborts **after** the model is built, not at config parse.
- `shardlib` dimension names are **global** and persist across `typed_shard_map`
  calls. Initializing an 8-layer target and a 2-layer draft in one process collides
  ("expected 8, got 2") unless each is wrapped in `with shardtypes.Scope():`.
- `typed_shard_map` derives partition specs from parameter *annotations*, so
  `Optional[X]` reaches `issubclass()` as a Union and raises. Every parameter needs
  a concrete shardtype annotation, including the return type.

---

## If you rebuild the dataset instead of restoring it

`tools/fetch_longcrawl64_subset.py` reproduces it in ~3 minutes, but
`clankur/longcrawl64` is a third-party mirror that can change. If it drifts, the
drafts would train on different data than the target did — quietly breaking the
comparison the whole project rests on. Check before trusting a rebuild:

```bash
cd ~/data && find . -type f | sort | xargs md5sum > /tmp/new.md5
diff /tmp/new.md5 ~/matX/spire_backup/dataset_manifest.md5 && echo IDENTICAL
```
