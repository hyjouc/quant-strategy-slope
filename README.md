# Frozen V18 baseline audit, 2020–2023

Read `baseline_audit_report.md` first. All times are UTC.

This is retrospective behavior auditing of a historically selected model, not independent out-of-sample validation. No V18 changes, model fitting, parameter optimization, or Step 2 work were performed. Funding is excluded.

The directory is an independent Git repository because its parent workspace was not a repository. `inputs/` contains unchanged frozen code/model snapshots and the strictly complete-closed-bar input. Original V18 files outside this directory are untouched.

Commands in the original workspace:

```powershell
python run_audit.py *> run.log
python validate_boundaries.py *> boundary_validation.log
python build_report.py *> report_build.log
```

The main script verifies source hashes against the original workspace `freqtrade_v18_baseline/manifest.json` and pre-2024 market input against `stage4_trade_attribution/manifest.json`. These source paths are intentionally required to recheck the original provenance; run from its existing location under `research/`. It does not use old performance results or precomputed features. A source hash mismatch or market range mismatch stops execution.

Python dependencies: numpy, pandas, pyarrow, matplotlib and scipy (Spearman correlation). Workspace `.research_deps` is supported; installed versions are in `environment_versions.json`. No network is needed.

`run.log` is the successful core computation log. `run_attempt1.log` preserves an earlier timezone comparison error fixed in the diagnostic code, before result delivery. `verification.csv` has 735 core checks. `boundary_validation.csv` has 17 additional tests for censored/pending trades, stop-touch ranges, fixed parameters, signal partitions and monthly reconciliation. `figure_validation.log` records regeneration after improving shared axes and color legends; analytical tables did not change.

Annual PNL is continuous marked-equity change, while closed-trade statistics use exit year. Side results are a decomposition of the same actual single-position portfolio, not separate directional backtests. Intrabar stop exit times are unknown; partial-bar highs/lows are excluded from MFE/MAE. One stop trade has no complete held bar and thus missing MFE/MAE/giveback rather than fabricated zero.

`artifact_manifest.json` hashes all deliverables except itself and Git bookkeeping. `git_delivery.txt` is generated after the commit to avoid a circular commit hash and is intentionally untracked. No remote push was made.
