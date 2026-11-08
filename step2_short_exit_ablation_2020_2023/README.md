# Step 2: short normal exit momentum qualification ablation

Read `step2_report.md`. All timestamps are UTC. The study only uses complete 2020–2023 candles, excludes funding, and is retrospective rather than independent out-of-sample evidence.

The sole experimental change is monthly-plan short `exit_kind: momentum -> sign`. The recorded `exit_value=0.38`, armed state, two confirmations, next-open fills and all other rules remain frozen. Both arms replay the entire continuous single-position portfolio.

Run from this directory, preserving its location under the Step 1 repository root:

```powershell
python run_step2.py *> run.log
python build_step2_report.py *> report_build.log
python validate_step2.py *> validation.log
```

The runner verifies input snapshots against Step 1 inputs before computing features, and exactly reconciles arm A to the original Step 1 trade table and directional/combined equity before starting B. Each arm is independently checked at zero slippage and at registered slippage; order sequences, fill prices and full equity curves must agree. `monthly_plan_A.csv` and `monthly_plan_B.csv` preserve every schedule field for inspection.

`run_audit.py` is a diagnostic helper copied from Step 1, with one additional diagnostic-only override to define B's full normal exit signal correctly. It is imported by `run_step2.py`; do not run its old Step 1 `main` entry point. The original Step 1 helper is not modified. Frozen strategy/model/replay files here are exact input snapshots. The independent scalar executor's two zero-slippage literals are replaced with 0.0002 only inside a temporary in-memory namespace for the original-cost validation; its file remains unchanged.

Dependencies: numpy, pandas, pyarrow, matplotlib. The existing workspace `.research_deps` is supported by the scripts; installed versions are recorded in `environment_versions.json`. Step 1 remains present in the parent directory and its 42 artifacts must remain byte-identical. `validate_step2.py` checks those artifacts and normal exits against real completed candles, armed state and next-open fills.

`run_initial.log` preserves the initial successful A/B computation. The final `run.log` rerun adds explicit input hash, fill-price and monthly-plan verification plus saved plans; there were no changes to either arm's trading rules, thresholds or results. The rerun was for verification and provenance, not a parameter experiment.

All differences in the main comparison tables use **B minus A**. Closed-trade statistics use exit year, while period PNL is the continuous marked equity difference. Side contributions belong to the same actual single-position portfolio. The matched-entry table is a descriptive cohort of actual shared entries, not a shadow portfolio. The report separately gives average normal-exit delay reduction and matched-entry actual exit advancement.

Intrabar stop exit timestamps are unknown; stop-touch candle extremes are excluded from complete-held-bar MFE/MAE. Normal-exit confirmation close and next candle open share the same clock boundary, so successful second-confirmation-to-fill waiting time is 0 minutes at 15m resolution; the actual next-open price and registered slippage still apply. No bar-close fill is assumed.

The report conclusion is **B has no clear advantage; retain A**. No deployment or Step 3 work was performed. The experiment branch is `research/v19-step2-short-exit-ablation-2020-2023`, based on Step 1 commit `03a71cf8c587bf7bf67b17aef7513fab3e6719e4`. The artifact manifest hashes this experiment directory's files, excluding itself and ignored Python caches. A post-commit delivery receipt is stored in the parent ignored `git_delivery.txt`.
