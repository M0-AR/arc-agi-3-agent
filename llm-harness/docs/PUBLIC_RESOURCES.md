# Public Kaggle resources this build depends on

All verified directly via the Kaggle API (not assumed) -- sizes, licenses,
and download counts as of the verification date. None of these are
Tufa-Labs-private or dfranzen-private resources; every one is public and
attachable by any Kaggle account.

## Datasets

| Ref | Size | License | What it actually is |
|---|---|---|---|
| `dfranzen/pennyroyal-v253` | 6.7 GB | Unknown (upstream-licensed components) | **Not model weights.** The offline SGLang serving wheelhouse for Qwen3.8-Flash-Next on the RTX Pro 6000, built from [John Pezzulli's Pennyroyal fork](https://github.com/jpezzulli/sglang-rtxpro6000) (release v2.5.3, commit `d00d88e`), plus 5 of Daniel Franzen's own patches (low-M BF16 GEMM, speculative-state memory-budget fix, Marlin scale-dtype fix, prefix-cache retention, bounded checkpoint prefetch). Confirmed by its own dataset description. 4,380 downloads. |
| `dfranzen/taaf-kaggle-source-bundle-copy` | 1.5 MB | Unknown | "Identical copy of Tufa Labs' original Duck source bundle" (`jeroencottaar/taaf-kaggle-source-share`), per its own description. Contains the pickled TAAF `Benchmark`/`deploy_target` objects plus the full `ARC3-Inference`/`tufa-arc-agi-framework` source tree under `src/`. This is the **base, unpatched** harness -- Franzen's actual ~500KB behavior patch is a separate file not included here (we deliberately run unpatched for this smoke test; see FINDINGS.md). 1,717 downloads. |

## Models

| Ref | Size | License | What it actually is |
|---|---|---|---|
| `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` | 181.2 GB | Qwen Community License 1.0 | Resharded (for faster Kaggle loading) copy of [Intel's Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound). Weights unchanged from upstream, including the BF16 PLE table; only the file layout changed. Mounts at `/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1`. |
| `dfranzen/albucino-qwen3-8-flash-next-drafter` | 4.2 GB | Qwen Community License 1.0 | MTP (multi-token-prediction) draft head extracted from [albucino's Qwen3.8-Flash-Next-W4A16-FP8PLE](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE), packaged separately for speculative decoding. Not a standalone LM. Mounts at `/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1`. |

## Competition data

| Ref | What it is |
|---|---|
| `competitions/arc-prize-2026-arc-agi-3` (note the `competitions/` prefix required by the classic `kernel-metadata.json` schema -- see FINDINGS.md Bug #1) | The competition's own bundled data: `arc_agi_3_wheels/` (the `arc-agi`/`arcengine` Python packages, no internet needed to install), `ARC-AGI-3-Agents/` (the official agent runtime), and `environment_files/` (all 25 public training games as real offline-playable environment files -- confirmed present: sk48, tn36, m0r0, bp35, cn04, dc22, tu93, lp85, ka59, wa30, vc33, lf52, r11l, sc25, sp80, ar25, sb26, cd82, re86, s5i5, ls20, ft09, su15, tr87, g50t). |

## Why these specific resources, and not Tufa's or dfranzen's actual private
ones

Tufa Labs' and dfranzen's real model/wheelhouse datasets referenced in their
own private Kaggle notebooks are not publicly attachable -- `get_notebook_info`
on their published notebooks shows the private ones as blank/inaccessible
source IDs. The resources above are ones Franzen (and, for the wheelhouse,
Pezzulli via Franzen's build) deliberately published as *separate, public*
datasets specifically so others could reproduce the Milestone 2 result --
confirmed by their own descriptions ("Full credit goes to...", explicit
reproduction instructions in `dfranzen/arc-agi-3-solution`'s `WRITEUP.md`).
Using them is exactly the reproduction path their authors intended, not a
workaround.
