# Cheese Pathogen Fate

Research code for forecasting the fate of *Escherichia coli* and *Staphylococcus aureus* in
traditional cheeses under dynamic temperature, pH and water-activity conditions, using an
explainable hybrid mechanistic–machine-learning model.

## Novelty in brief

| Direction | Question | Code |
|---|---|---|
| A. Forecast mode | Predict pathogen fate without future pH, aw and LAB measurements; quantify the gap to measured-input models | `forecast.py` |
| B. Deacidification | Does the pH rise during ripening slow pathogen die-off? Tested with SHAP | `curation.ph_rebound` |
| C. Time-to-safe | Ripening time to a target reduction with a conformal coverage guarantee | `conformal.py` |

## Repository layout

```text
cheeses/
├── data/
│   ├── raw/combase_cheese/     # ComBase exports (not tracked; see its README)
│   ├── processed/cheese/       # generated observation tables (not tracked)
│   └── lab_links_verified.csv  # verified LAB links for the Ec/Sa study
├── scripts/
│   ├── download_combase.py     # re-download the ComBase exports
│   ├── build_dataset.py        # exports -> observation tables + data audit
│   ├── run_baseline.py         # mechanistic baseline, grouped CV
│   ├── run_experiments.py      # all models x input modes x origins, grouped CV
│   └── analyze_experiments.py  # RQ1-RQ6 tables, SHAP test, time-to-safe, figures
├── src/cheese_fate/
│   ├── combase.py          # load ComBase Excel exports into tidy tables
│   ├── curation.py         # align conditions and same-batch LAB, audit, pH rebound
│   ├── mechanistic.py      # dynamic Baranyi + cardinal models + inactivation, fitting
│   ├── organisms.py        # cardinal presets with provenance
│   ├── baseline.py         # grouped CV of the mechanistic baseline
│   ├── tasks.py            # prediction tasks, input modes, leakage-safe features
│   ├── models.py           # persistence, mechanistic, censored GBT, residual hybrid
│   ├── sequence.py         # GRU/LSTM and neural-ODE hybrid (PyTorch)
│   ├── coupled.py          # coupled LAB -> pH -> aw forecaster (direction A)
│   ├── experiment.py       # per-fold training and prediction of every model and mode
│   ├── analysis.py         # metrics, forecast gap, LAB effect, SHAP test, time-to-safe
│   ├── figures.py          # report figures
│   ├── forecast.py         # forecast-mode input masking (direction A)
│   ├── conformal.py        # record-level conformal bands, time-to-safe (direction C)
│   └── evaluation.py       # metrics, leave-one-group-out and grouped folds
├── tests/
├── models/                 # local model artifacts (not tracked)
└── reports/                # local figures and metrics (not tracked)
```

Everything the study needs lives in this folder:

- `data/raw/combase_cheese/exports/`: 229 ComBase exports, 4561 records (see that folder's
  README).
- `data/processed/cheese/`: generated observation tables.

The raw data can be re-downloaded with `scripts/download_combase.py`.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,models,explainability,figures]"
pytest
```

## Build the dataset

From this folder:

```bash
python scripts/build_dataset.py
```

This writes:

- `reports/data_audit.csv`: data availability per organism;
- `data/processed/cheese/observations_escherichia_coli.csv` (297 curves, 2835 observations);
- `data/processed/cheese/observations_staphylococcus_aureus.csv` (223 curves, 2070
  observations).

Each row is one measured log count with aligned temperature, pH, aw and same-batch LAB count.
Every aligned value that was not measured at that exact time has an `*_interpolated` flag.
Counts are never interpolated.

## Run the experiments

From this folder (about 3 min per organism on the source split, and about 10 min on
the cheese-type split, which refits the mechanistic model):

```bash
python scripts/run_baseline.py                      # mechanistic baseline, fold fits
python scripts/run_experiments.py --group source_id # primary split
python scripts/run_experiments.py --group food      # unseen cheese types (RQ6)
python scripts/analyze_experiments.py               # tables, tests, figures
```

Outputs go to `reports/experiments/<group>/`: `summary.md` (all RQ tables), CSV tables and
`figures/`. Each `*_run.json` records the split, seeds, configuration and per-fold fit details.

## Reproduce the manuscript

The exact package versions are in `requirements.txt`. From this folder, after the
steps above:

```bash
# robustness: two further split seeds and leave-one-source-out
python scripts/run_experiments.py --group source_id --seed 1 --name source_id_seed1
python scripts/run_experiments.py --group source_id --seed 2 --name source_id_seed2
python scripts/run_experiments.py --group source_id --folds 0 --name source_id_loso
python scripts/run_cardinal_sensitivity.py          # Table S1 variants

# LAB-mapping sensitivity (Section 3.3): verified links, and the Ec/Sa source left out
python scripts/run_experiments.py --group source_id --name source_id_lablinks \
    --lab-links data/lab_links_verified.csv
python scripts/run_experiments.py --group food --name food_lablinks \
    --lab-links data/lab_links_verified.csv
python scripts/run_experiments.py --group source_id --name source_id_noEcSa --exclude-sources Ec Sa
python scripts/run_experiments.py --group food --name food_noEcSa --exclude-sources Ec Sa

# bootstrap intervals and paired tests, then figures and supplementary tables
python scripts/run_inference.py --names source_id food source_id_seed1 source_id_seed2 \
    source_id_loso source_id_lablinks food_lablinks source_id_noEcSa food_noEcSa
python scripts/summarize_robustness.py
python scripts/make_publication_figures.py
python scripts/make_supplementary_tables.py
```

Figures go to `reports/figures/` and supplementary tables to `reports/supplementary_tables/`
(pass `--out` to write them elsewhere).

`data/lab_links_verified.csv` lists the 20 pathogen records of the Ec/Sa study whose
LAB record could be verified: same treatment and replicate in the record name, and identical
pH, aw and temperature series (a unique match). The study's other pathogen records match only a
non-inoculated control or several LAB records, so they stay without LAB.

## Data and licence

The code is released under the MIT licence (`LICENSE`); cite it with `CITATION.cff`. The
ComBase records are not redistributed here: download them from ComBase
(<https://www.combase.cc>) under its terms of use, either by hand or with
`scripts/download_combase.py`, into `data/raw/combase_cheese/exports/`.

## Current status

- **Done and tested (37 tests):**
  - data download, parsing and curation;
  - the mechanistic baseline;
  - ML-only models: censored GBT, GRU and LSTM;
  - hybrids: residual GBT and neural ODE, each with a LAB ablation;
  - the coupled LAB → pH → aw forecaster;
  - the experiment runner and analyses: RQ1–RQ6, SHAP test, cross-conformal time-to-safe,
    figures.
- **Results** (5 folds, origin 0, pooled RMSE in log10 CFU/g on held-out detected counts;
  persistence is the reference to beat). Details are in
  `reports/experiments/<group>/summary.md`.

  | Split | Organism | Persistence | Mechanistic | Best ML-only | Best hybrid |
  |---|---|---|---|---|---|
  | by source | *E. coli* | 2.10 | 2.29 | **1.78** GRU (beats persistence in 4/5 folds) | 1.95 residual GBT |
  | by source | *S. aureus* | **1.64** | 3.07 | 1.78 GBT | 2.21 residual GBT |
  | by cheese type | *E. coli* | 2.10 | 2.45 | 1.85 GRU | **1.78** residual GBT (5/5 folds) |
  | by cheese type | *S. aureus* | **1.64** | 2.21 | 1.73 GBT | 1.73 residual GBT |

  - **RQ1:**
    - *E. coli* is predictable across sources and cheese types: 12–15% lower RMSE than
      persistence.
    - *S. aureus* is not: no model beats persistence with measured inputs.
    - The neural-ODE hybrid does not beat persistence, because it inherits the mechanistic
      model's error.
  - **RQ2:**
    - Forecast mode costs almost nothing. With pH, aw and LAB held at their values at the
      origin (`naive`), accuracy equals or beats measured inputs, e.g. *E. coli* GRU 1.77 vs
      1.78. For *S. aureus*, the naive GRU (1.48) even beats persistence.
    - Measured future trajectories therefore carry patterns that do not transfer across
      sources.
    - The coupled forecaster predicts pH and LAB far better than holding the last value from
      day 0 (pH RMSE 0.50 vs 1.18; LAB 0.89 vs 2.33 log for *E. coli*). It is no better from
      day 7 or day 30, and it does not improve the pathogen forecasts.
  - **RQ3:** same-batch LAB helps the residual hybrid (*E. coli*: −0.14 log, better in 4/5
    folds; *S. aureus* coupled mode: 5/5 folds). The effect for plain GBT is mixed.
  - **RQ4 (pre-registered): not supported.** After the pH minimum, the pH attribution is not
    positive. It *decreases* with the rebound size (*E. coli* GBT ρ = −0.32 in rebound curves;
    the one-sided test for an increase gives p ≈ 1).
  - **RQ5:**
    - Cross-conformal bands reach their nominal 90% coverage (0.87–0.91), but they are wide:
      about ±4 log two-sided, and +3.4 / +2.9 log one-sided over ripening.
    - No curve can be certified safe at 90%. Even in the observed data, only 61% / 62% of the
      curves followed past day 60 stay at or below the detection limit from day 60 on.
- **Next:**
  - write up the results;
  - decide whether to report the RQ2 finding (naive ≥ measured) as a main result, and whether
    to probe it (e.g. pH/LAB measurement differences between sources);
  - optionally, finer folds (leave-one-source-out).
- **Open:** confirm the provisional *E. coli* cardinal values and the pH limits from the full
  texts (`organisms.py`).
