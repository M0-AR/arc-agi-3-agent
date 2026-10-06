# Findings: getting a Duck-harness notebook to actually mount data and run

Same discipline as [`../../docs/FINDINGS.md`](../../docs/FINDINGS.md): every
bug below was found by reading real evidence from an actual Kaggle kernel
run, not guessed, and nothing was considered fixed without the next real run
proving it.

All runs below are against the kernel `mohamadaiashmar/arc3-seif-attach-test`
(or, briefly, a since-abandoned sibling, `arc3-smoke-test-clean-kernel` --
see Bug #3) on Kaggle, pushed via different tool paths while debugging which
one actually works.

## Bug #1 -- the `save_notebook`-style API path never mounts anything

**Symptom:** a notebook built to install `arc-agi` from the competition's
bundled wheels, then read a dataset's `requirements.lock`, failed
immediately:

```
WARNING: Location '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels' is ignored: it is either a non-existing path or lacks a specific scheme.
ERROR: Could not find a version that satisfies the requirement arc-agi (from versions: none)
```

**Evidence:** a minimal diagnostic cell (`ls -la /kaggle/input; find
/kaggle/input -maxdepth 5`) showed `/kaggle/input` completely empty --
`total 8`, just `.` and `..` -- across **three separate attempts**, each
with different `competitionDataSources`/`datasetDataSources` string formats
(bare slug, prefixed, combined with datasets+models). Fetching the pushed
kernel's own metadata back confirmed the push tool's explicit
attachment parameters were never even being recorded: no
`competition_data_sources` field appeared in the kernel's metadata at all,
where a known-working sibling kernel in this same account
(`arc3-step-4-learned-path-value`, built via a different, working push path
earlier in this project) *did* show `"competition_data_sources":
["arc-prize-2026-arc-agi-3"]`.

**Fix:** stopped using that API-parameter-based push path entirely. Switched
to a classic, file-based `kernel-metadata.json` + folder push (the `seif-kaggle`
MCP server's `kernel_push`, which mirrors the real `kaggle kernels push -p
<folder>` CLI exactly). Confirmed working immediately: a fresh kernel pushed
this way showed real files under `/kaggle/input/competitions/.../arc_agi_3_wheels/`
on the very next run.

## Bug #2 -- wrong mount-path convention assumed for datasets

**Symptom:** even after fixing attachment, code written against the paths
used in Daniel Franzen's own real submission notebook --
`/kaggle/input/datasets/dfranzen/pennyroyal-v253` -- found nothing there.

**Evidence:** a `find /kaggle/input -maxdepth 6` on a run that *did* have
the dataset attached showed it actually mounted at
`/kaggle/input/pennyroyal-v253` directly -- no `datasets/` segment, no
`dfranzen/` owner prefix. Models, by contrast, mounted exactly where expected:
`/kaggle/input/models/dfranzen/<slug>/transformers/default/1`.

**Fix:** corrected `WHEELHOUSE_DIR` and `ORIG_BUNDLE_DIR` in the harness
setup cell to `/kaggle/input/pennyroyal-v253` and
`/kaggle/input/taaf-kaggle-source-bundle-copy` respectively. (Franzen's own
notebook likely mounts differently due to how *his* Kaggle account's dataset
versioning/owner history resolved at push time -- not a universal path, as
this discrepancy proves.)

## Bug #3 -- stale per-kernel attachment state on a kernel with a troubled
push history

**Symptom:** with the correct paths and a working push mechanism, one
kernel (`arc-agi-3-baseline-smoke-test`, which had accumulated 7 versions
across the Bug #1/#2 debugging, including several pushed via the broken
API-parameter path) intermittently failed to mount the `pennyroyal-v253`
dataset specifically -- 3 times in a row, with the *exact same*
`kernel-metadata.json` that mounted it correctly on a different, cleanly-
created kernel (`arc3-seif-attach-test`).

**Evidence:**
```
RuntimeError: /kaggle/input/pennyroyal-v253 never mounted (missing requirements.lock) after 600.0s
```
...on `arc-agi-3-baseline-smoke-test`, three separate times (once at a
180s timeout, twice at 600s), while the identical dataset mounted correctly
and immediately (`Precaching: loaded 211 files (6.70 GB) in 9.55 seconds`)
on `arc3-seif-attach-test`.

**Fix:** abandoned the troubled kernel; retargeted all further pushes at the
kernel with a clean attachment history. Added a `_wait_for_mount()` guard
(poll for a known file inside each mounted path, with a generous timeout)
as defense-in-depth regardless, since Kaggle's dataset mounting is
asynchronous relative to kernel startup and *can* legitimately lag --
this is now a permanent, cheap safety net even on a clean kernel.

## Bug #4 -- Python 3.13 vs. the wheelhouse's pinned cp312 wheels

**Symptom:** with mounting fully fixed, the Pennyroyal install itself failed:

```
error: No solution found when resolving dependencies
  cause: Because aiohttp==3.14.3 has no wheels with a matching Python ABI tag (e.g., `cp313`) and you require aiohttp==3.14.3, we can conclude that your requirements are unsatisfiable.
hint: You require CPython 3.13 (`cp313`), but we only found wheels for `aiohttp` (v3.14.3) with the following Python ABI tag: `cp312`
```

**Cause:** Kaggle's current GPU-image Python is 3.13, but
`dfranzen/pennyroyal-v253`'s `requirements.lock` was pinned to `cp312`
wheels when Franzen built it (consistent with Kaggle's base image having
since moved forward). The harness's own venv-creation step used
`sys.executable` (the notebook's own 3.13 interpreter) unconditionally.

**Fix:** before creating the venv, look for an exact `python3.12` binary on
the image (`shutil.which('python3.12')` / `/usr/bin/python3.12`) and prefer
it over `sys.executable` if present. Verified immediately on the next run:

```
venv interpreter: /usr/bin/python3.12 | system sys.executable: /usr/bin/python3
$ /tmp/sgl-intel/bin/uv venv --python /usr/bin/python3.12 /tmp/sgl-intel/venv
```

...and the full install then succeeded completely, including the actual
SGLang wheel (`sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl`).

## Open issue (unresolved, not a code bug) -- no GPU actually attached

**Symptom:** after Bug #4's fix, the notebook ran cleanly all the way to a
CUDA check, which then failed:

```
RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
```

**Evidence:** a direct diagnostic (`nvidia-smi`) confirmed the command
doesn't exist on the machine at all:
```
FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi'
```
This is despite the kernel's own fetched-back metadata correctly showing
`"enable_gpu": true` and `"machine_shape": "NvidiaRtxPro6000"` -- confirmed
identically across **both** push mechanisms (the broken API-parameter path
from Bug #1 *and* the working file-based path), and the kernel's
`docker_image` staying on the CPU-only
`gcr.io/kaggle-images/python@sha256:81b1e...` rather than the GPU-specific
`gcr.io/kaggle-gpu-images/python` image in every single run regardless of
push method.

**Diagnosis:** matches Kaggle's own documented behavior that GPU/TPU access
requires phone number verification on the account, and that an unverified
account's GPU request is accepted into the kernel's stored configuration but
silently never actually provisioned at runtime -- exactly the symptom
pattern observed.

**Re-tested after reported phone verification -- still unresolved.** A
fresh push (same `kernel-metadata.json`, same notebook, nothing else
changed) gave an *identical* result: `"enable_gpu": true` and
`"machine_shape": "NvidiaRtxPro6000"` both correctly recorded on the kernel,
`docker_image` still `gcr.io/kaggle-images/python@sha256:81b1e...` (the
CPU-only image), and the same
`FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-smi'`.
So either the verification hasn't finished propagating on Kaggle's side, the
account's verification didn't actually complete the way it looked like it
did, or phone verification isn't the whole story here. This needs
checking directly on kaggle.com (open a notebook through the website UI
and try switching its Accelerator dropdown to the RTX Pro 6000 by hand) to
tell whether the gap is account-level or specific to pushing via the API.

## Where this leaves it

Confirmed working end-to-end on CPU: competition data, both datasets, and
both models all mount correctly on a clean kernel; the full offline
Pennyroyal/SGLang install completes successfully against a Python 3.12 venv.
**Not yet confirmed**: that a real GPU is actually attached and the SGLang
server actually starts and serves the model. GPU provisioning remains
blocked after one retest; the next step is verifying GPU access directly
through the Kaggle website UI (not the API) to isolate whether this is an
account-verification gap or something specific to API-driven kernel pushes.
