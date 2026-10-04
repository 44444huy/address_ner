# Evaluation data policy

`golden_cases.seed.jsonl` was used to inspect model errors before targeted hard-case
training. It is therefore a development set, not an unbiased final test set.

The code still calls it a golden dataset for historical compatibility, but its score
must be reported as a development score.

`resolver_cases.seed.jsonl` is also a development set. It contains manually selected
exact, typo, ambiguous, invalid and hierarchy-conflict cases used to calibrate the
resolver thresholds. Report `reports/resolver.calibration.seed.json` as a development
calibration result, never as final production accuracy.

A final holdout must be created separately with these rules:

1. Use 100-300 real or manually reviewed addresses.
2. Include current, legacy, abbreviated, unaccented, partial, ambiguous and invalid input.
3. Do not copy exact text from synthetic or training datasets.
4. Freeze the file before final model comparison.
5. Do not inspect per-case model errors and then train against the same holdout.
