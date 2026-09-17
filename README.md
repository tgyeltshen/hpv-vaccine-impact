# HPV vaccination impact: analysis code

Analysis code for a pre-specified global evaluation asking whether modelled cancer incidence
estimates, combined with national programme schedules, can identify the population effect of
human papillomavirus vaccination on invasive cervical cancer.

**The answer reached was no.** The study was computationally feasible but causally
unidentified. A staggered difference-in-differences design returned an apparent protective
association that reappeared at equal or greater magnitude in placebo analyses containing no
post-eligibility observations. Four barriers were each independently sufficient to invalidate
the design: no valid comparison group, extreme concentration of information in a single
country, immature cohort follow-up, and an outcome that is not neutral to programme status.
Conventional diagnostics, including a passing pre-trend test, concealed all four.

The pipeline is built to fail loudly rather than quietly: `run_pipeline.py --stage analyse`
applies the protocol's six pre-specified decision rules to its own output and **exits non-zero
when those rules forbid a causal conclusion**. A green pipeline never implies a positive
finding.

## What is in this repository

Only the analysis code, under `04_code/`.

```
04_code/
  run_pipeline.py            single restartable entry point
  download_sources.py        provenanced retrieval of every source, with checksums
  src/
    ingest.py                source loading
    exposure.py              cohort eligibility from WHO/UNICEF schedules
    covariates.py            country covariates
    build_pipeline.py        analytic panel construction
    panel_models.py          group-time and interaction-weighted estimators
    analysis.py              estimation, falsification tests, decision rules
    maturity.py              cohort maturity accounting
    power.py                 minimum detectable effect
    ci5_validation.py        validation against IARC CI5
    cancer_over_time.py      validation against IARC Cancer Over Time
    estimation_report.py     results assembly
  tests/                     9 test modules
```

## What is not in this repository, and why

The data directories are absent by design. Raw source archives are third-party and their
licences do not permit redistribution; the derived panels, results, figures and manuscript
drafts are not published here either.

This matters for running the code. The scripts resolve the project root as the parent of
`04_code/` and expect these sibling directories to exist:

```
01_metadata/    source manifest, dictionaries, query specifications
02_raw_data/    immutable source bytes, grouped by source and release
03_processed_data/
05_results/  06_figures/  07_tables/  09_logs/
```

`download_sources.py` retrieves the public sources and writes `02_raw_data/` together with the
manifest that `run_pipeline.py --stage validate` checks. Every source is public and free of
charge: Global Burden of Disease 2023, UN World Population Prospects 2024, WHO and UNICEF
electronic Joint Reporting Form schedules via WHO's public xMart4 OData endpoints, the WHO
Global Health Observatory, World Bank indicators, and IARC Cancer Incidence in Five Continents
and Cancer Over Time. No endpoint used here requires credentials.

## Running it

```bash
python -m pip install -r requirements.txt

python 04_code/download_sources.py            # retrieve sources into 02_raw_data/
python 04_code/run_pipeline.py --stage validate   # checksums and manifest
python 04_code/run_pipeline.py --stage build      # construct the analytic panel
python 04_code/run_pipeline.py --stage analyse    # estimate; non-zero exit is by design
python 04_code/run_pipeline.py --stage all

python -m pytest 04_code/tests -q
```

Optional flags on the analyse stage: `--uncertainty-draws N` propagates the modelled-outcome
interval into the overall ATT, and `--no-leave-one-out` skips the leave-one-country-out refits.

## Citation

This code accompanies a manuscript under submission. Please cite the article once published.

## Licence

MIT. See [LICENSE](LICENSE).

This covers the code in this repository only. The data sources it retrieves are third-party
and carry their own terms; see the provider in each case.
