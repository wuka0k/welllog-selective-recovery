# Well-log selective recovery

This repository contains the custom analysis code for the manuscript
**Recoverability-bounded selective cross-log recovery for whole-profile missing
well logs**. The code compares SAITS with a whole-profile cross-log expert,
constructs pattern-specific conformal intervals, applies the
accept/recover/defer rule, and evaluates the recovered logs in a lithology
classification audit.

## Quick test

The operator smoke test uses only the Python standard library:

```bash
python example_quick_test.py
```

Expected output:

```text
quick test passed
```

## Environment

Python 3.10 or newer is recommended. The reported experiments used Python
3.14, PyTorch 2.15, and one NVIDIA RTX 3090 Ti.

```bash
python -m pip install -r requirements.txt
```

## Data and upstream benchmark

The public benchmark code is not redistributed here because its repository does
not currently declare a license. Clone it as a sibling directory:

```bash
git clone https://github.com/uai-ufmg/well-log-imputation.git well_log_imputation
git -C well_log_imputation checkout e69031ad97e19f6e36c0dddfc3db68ca2ce03716
```

Download the processed Teapot, Taranaki, and Geolink datasets from Zenodo DOI
`10.5281/zenodo.10987946` and extract them so that
`well_log_imputation/preprocessed_data/pgeoprj/godeep/dados/l2_datasets/imputation-processed-datasets/`
contains the fold `.npy` and metadata files.

Archive SHA-256:

```text
67A659043770FAB643D60D4D8FD674DA50D178797F49E664D8C9ABDC532A7AA6
```

The downstream audit uses FORCE 2020 from:

```text
https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
```

Place `train.zip` in `force_data/train.zip` next to the directories containing
this repository and `well_log_imputation`, or adjust the path in
`run_lithology_downstream.py`.

## Reproduce the recovery analysis

Run from this repository:

```bash
python run_recovery_experiment.py \
  --datasets teapot taranaki geolink \
  --seeds 5 --masks-per-seed 5 \
  --epochs 100 --patience 15 --suffix mid100
```

Fold-specific runs use `--fold 1`, `--fold 2`, `--fold 3`, or `--fold 4` and a
distinct `--suffix`, for example `--suffix fold1_geolink`. The aggregation
script reads the fold-0 file and the remaining fold files:

```bash
python make_full5fold_analysis.py
```

The model-family control can be rerun with:

```bash
python run_profile_baselines.py
```

Set `WELLLOG_BENCHMARK_ROOT` if the benchmark is not a sibling directory.
Set `WELLLOG_EVIDENCE_ROOT` to write outputs to an explicit directory.

## Downstream audit

```bash
python run_lithology_downstream.py
python make_downstream_summary.py
```

The downstream script requires the FORCE archive described above.

## Regenerate manuscript figures and tables

The frozen evidence files used in the paper are included so that the reported
figures and LaTeX tables can be regenerated without training a model:

```bash
python manuscript_assets/analyze_thresholds_and_bounds.py
python manuscript_assets/make_assets.py
```

The first command rebuilds threshold-sensitivity and identifiability summaries.
The second writes `figures/fig2_profile_recovery.*`,
`figures/fig3_mechanism_boundary.*`, `figures/fig4_downstream.*`, and the
manuscript tables under `tables/`. The scripts require the packages listed in
`requirements.txt` and consume only files in `evidence/`.

## Outputs

The recovery runner writes:

- raw per-seed, per-mask and per-feature results;
- condition-level summaries;
- paired Wilcoxon tests with Holm correction;
- model and calibration manifests.

The downstream runner writes:

- per-split raw-missing, mean-fill, cross-log and oracle metrics;
- summary distributions;
- paired split-level tests.

The manuscript asset scripts write publication figures and LaTeX tables. These
outputs are generated from frozen evidence; no numeric points are entered by
hand.

## License

The custom scripts in this repository are released under the MIT License.
The external benchmark source and external data retain their own terms.

## Contact

Bingqian Hao  
Haobingqian129@outlook.com  
ORCID: 0009-0007-6312-2944
