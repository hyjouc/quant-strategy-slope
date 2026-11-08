# Step 4 reproducibility

Run from this directory using Python 3.14.3 with numpy, pandas, pyarrow, matplotlib and Pillow. The workspace fallback `.research_deps` is loaded by the runner. Installed versions are recorded in diagnostic_metadata.json.

```powershell
python run_step4.py *> run.log
if ($LASTEXITCODE -ne 0) { throw 'Replay failed; stop' }
python finalize_step4.py *> delivery_validation.log
if ($LASTEXITCODE -ne 0) { throw 'Validation failed; stop' }
python manifest_step4.py
```

Requires unchanged Step 1 inputs, Step 2 helpers and Step 3 artifacts in the parent repository at the declared base ancestry. Only A and B are run. Initial baseline branch/head/remote preflight was performed before branch creation; metadata initial_head records the exact Step 3 SHA. Hash checks protect all original artifacts. Finalization independently reconstructs actual-entry masks/scores, validates table accounting, and renders eight figure sets from the delivered tables. It does not train, search, or replay other experiments.

Manifest excludes itself, caches and .pyc. Commit/remote receipt is outside the hashed experiment to avoid self-referential hashes. Read task_request.md for the full registered protocol and step4_report.md for findings and limitations.
