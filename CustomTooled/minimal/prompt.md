# Building System ID

Train a physics-informed neural network on the building data in `data/split/`. Edit only `src/user_defined.py`, `src/configs/*.yaml`, `data/grad/*.json`, `data/measurement_report.md`, and `final_report.md`. Don't touch other `src/` files.

1. Map data columns to states (Y), controls (U), and disturbances (D) in `user_defined.py`. Train a baseline with no physics losses. Confirm it works.
2. Diagnose what the model gets wrong. Iterate: adjust config, optionally add physics constraints (gradient signs, synthetic scenarios), change architecture — whatever the diagnosis calls for. Explain each change and why. No automated hyperparameter search.
3. After maximum 5 iterations, or when performance plateaus or is satisfactory, stop and create `final_report.md` documenting each iteration: what you tried, what metrics resulted, and why you made each change.

Run from `src/`: `uv run python main.py --config <name>`. Override params with `--set key=value`.
