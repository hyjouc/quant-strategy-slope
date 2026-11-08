# Step 3: selection bias and robustness audit

Read `step3_report.md`. This is retrospective conditional evidence, not independent out-of-sample validation. No models or thresholds are searched or fit. Final cap recipe selection historically used 2023–2024; all current market input is strictly pre-2024 complete candles.

Run in the existing workspace layout:

```powershell
python run_step3.py *> run.log
python build_report.py *> report_build.log
python validate_step3.py *> validation.log
```

The script imports the already committed Step 2 experiment helper. It reads prior Step 1/2 complete trade/equity outputs and Step 1's verified `inputs/ohlcv_closed_pre2024.feather`. The historical optimizer scripts are read as **text only**, never executed/imported. Source paths and snapshots appear in `evidence_index.csv`. The only JSON selected from historical model output contains coefficient/scale/training metadata, not market rows or post-2023 performance results. Read-only historical sources outside the repository are needed to repeat provenance collection; the delivered snapshots suffice to review its evidence.

Main outputs:

- `parameter_provenance.csv`: actual values, documented selection windows, evidence and UNKNOWN attempt counts. Missing anchors/steps are explicitly recorded.
- `common_normal_short_ranked.csv`, `concentration_summary.csv`, `concentration_deletions.csv`: original238-trade cohort, A-better58/B-better108/equal72; descending A−B contributions, same-entry paired deletions. These are diagnostic ledgers, never alternate formal strategy returns.
- `time_stability_all_windows.csv`: all444 annual/quarter/month/rolling6m/rolling12m direction rows. PNL uses marked change, closed-trade PF/count use exit time; no window starts with a forced position reset. `time_stability_summary.csv` summarizes all windows.
- `daily_marked_pnl.csv`, `bootstrap_summary.csv`, `bootstrap_paired_replicates.csv.gz`: identical-index paired overlapping noncircular moving-block sampling, 28/56 consecutive UTC days,20000 repetitions each. Last block truncates to1452 days. Bootstrap does not repair historical strategy-selection bias.
- `cost_stress_all.csv`, `cost_stress_comparison.csv` and per-scenario trade/equity files: total1×/1.5×/2× simultaneous fee+slippage multipliers, entire portfolio replayed and independently verified; indicator/gate formulas and all strategy logic fixed. Costs can change fill-anchored stops/early exits and thus later execution paths.
- `fixed_ledger_cost_sensitivity.csv`: additional accounting-only fixed-original-trade repricing, explicitly not a lawful full portfolio simulation under new cost-anchored stops.
- `verification.csv`, `validation.log`, `diagnostic_metadata.json`, figures and `artifact_manifest.json`: executed assertions, provenance and review material.

`run_attempt1.log` preserves a calendar endpoint timezone error; `run_attempt2.log` preserves an independent verifier rejection caused by a diagnostic string substitution of the fee literal. Both diagnostic defects were corrected without changing any strategy parameter or outcome selection; final `run.log` records all successful cost scenario reconcilations. No failed computation is used in delivered results. The only arms are inherited A/B and only cost multipliers are1,1.5,2.

The parent Step 1 and Step 2 artifacts remain byte-identical, checked against their committed manifests. Dependencies use existing numpy/pandas/pyarrow/matplotlib with optional workspace `.research_deps`; versions are saved. No network is needed for computation. The study stops here for researcher review.
