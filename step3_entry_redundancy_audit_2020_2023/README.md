# Step 3: entry redundancy and independent information audit

This directory contains only signal-level diagnostics. It neither runs a complete strategy ablation nor changes a trading condition. Read `step3_report.md` and its question-by-question conclusions first.

Branch: `research/v19-step3-entry-redundancy-audit-2020-2023`, based on Step 2 commit `c2cedb8524ebc7c6574c03d41cb0decdbaf086a5`. Step 1 and Step 2 remain unchanged. The unrelated pre-existing untracked `step3_robustness_audit_2020_2023/` is preserved and not read, committed or used as evidence in this task. That exact directory is excluded only through local `.git/info/exclude`, so it does not pollute this branch's delivery status; no user files were deleted or hidden by a committed ignore rule.

Run in the existing repository layout:

```powershell
python run_step3.py *> run.log
python present_step3.py *> report_build.log
python validate_step3.py *> validation.log
```

The runner imports Step 1's frozen feature/scoring helpers. It reads only the hash-verified pre-2024 closed-bar input and frozen registration/model records; it never calls the trading replay. It verifies Step 1 historical counts rather than generating new strategy performance. All major comparisons use hypothetical next-open entry prices, no fees/slippage, no leverage or position constraints. They are overlapping signal-level price diagnostics, not investable portfolio returns.

The delivered computation first generated 384 comparisons, then explicitly added 24 common-clock confirmation comparisons with `common_clock_confirmation.py` using the same saved bootstrap draws, for 408 comparisons total and 816 rows across two block lengths. This controls the mechanical inclusion of the confirming candle in first-signal success/failure grouping. The final main script includes these same 24 definitions, so a complete rerun needs only the commands above; the supplementary script is preserved to document the actual execution path and is idempotent. Its successful output is in `common_clock.log` and appended to the final `run.log`.

All four technical gates remain in the actual strategy. A candidate mask with one gate omitted is a diagnostic counterfactual only. The pipeline masks still retain the original Ridge feature values, weights, scaling, clipping and threshold; removing an explicit M/S/A gate does not remove that information source from Ridge. Extra two-confirmed candidate bars are separately counted, with original month-boundary eligibility, but are not actual additional account trades.

All three registered Ridge models are decomposed; `filtered_long` is contextual only, not a substitute baseline or new experiment. Feature contributions are descriptive `beta * clip(z, -5, 5)` and do not establish causal feature importance. Linear nuisance projections for partial correlations do not refit the frozen predictive model.

Bootstrap uses common non-circular moving calendar blocks of 28 or 56 days, 10000 draws each. All signals belonging to a sampled calendar day are kept together. Saved draws make resampling exactly reproducible. Confidence intervals concern **mean differences**, not future signal predictions. Group medians are point estimates; no median confidence interval is implied. In the paired confirmation comparison, group1 stores each successful event's second-minus-first price diagnostic and group0 is a zero reference on the same events. Raw first/second outcome summaries are in `confirmation_outcomes.csv`.

`run_attempt1.log` records the initial EMA-availability sample-count assertion failure, corrected by preserving frozen missing-value handling. `run_attempt2.log` records an interrupted performance-heavy computation. Correlation partials were then calculated using the inverse correlation matrix (same linear partial correlation), instead of repeatedly residualizing each feature pair. No thresholds, sample definitions, outcome formulas or results were tuned. `run.log` is the completed calculation log; `validation.log` independently reconstructs selected bootstrap intervals using daily cumulative sums rather than the runner's day-weight matrix.

Dependencies: numpy, pandas, pyarrow, scipy, matplotlib, Pillow; the workspace `.research_deps` is supported. `artifact_manifest.json` covers this directory's deliverables except itself, caches and delivery receipts. `diagnostic_metadata.json` records sources, boundaries, seeds and verification. The study is exploratory and historically selected; 2023 is not independent out-of-sample evidence. Neither bootstrap nor partial correlations remove historical selection bias, multiple comparisons or causal confounding.

No deployment, model training, threshold search, strategy upgrade or next-stage ablation was performed.
