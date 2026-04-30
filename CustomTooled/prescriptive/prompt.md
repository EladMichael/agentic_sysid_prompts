# System Identification Framework — Iterative Training & Refinement

You are adapting a physics-informed neural network (PINN) framework to a new building dataset. Your job is to bridge between the user's data and the system identification backend. You will iteratively build, train, and refine models — starting simple and layering in complexity only when justified by results. Your work should read as a clear engineering narrative: at every stage, explain what you observed, what you changed, and why.

---

## Files You Edit

- `data/measurement_report.md` — dataset documentation
- `data/grad/*.json` — gradient sign constraint mappings
- `src/user_defined.py` — column mappings, synthetic configs, gradient templates
- `src/configs/*.yaml` — training configurations
- `final_report.md` — summary report documenting all iterations, decisions, and results

## Files You Never Touch

All other files in `src/` are framework internals: `main.py`, `optuna_main.py`, `loss.py`, `models.py`, `experiment.py`, `data_utils.py`, `gradient_sampling.py`, `synthetic_scenarios.py`, `plotting.py`, `config.py`, `load_model.py`.

---

## Conventions

- **Y** = state outputs the model predicts (e.g., zone temperatures)
- **U** = controllable inputs (e.g., setpoints, flows, valve positions)
- **D** = uncontrollable disturbances (e.g., weather, occupancy)
- **Config cascade**: `Config` dataclass defaults < YAML file < CLI `--set` overrides
- **CLI overrides**: `--set key=value` (repeatable) on any run command

---

## How to Run

All commands run from `src/`:

```bash
cd src

# Single training run
uv run python main.py --config <config_name>

# Load and evaluate a trained model
uv run python load_model.py --run_id <MLFLOW_RUN_ID>

# View experiment results
uv run mlflow ui
```

Config names can be a short name (e.g., `mlp_noPINN`) or full path (e.g., `configs/mlp_noPINN.yaml`). Override any parameter with `--set`, e.g., `--set batch_size=64 --set epochs=200`.

**Do not use Optuna or automated hyperparameter search.** All tuning should be done manually through deliberate, reasoned changes between iterations.

---

## General Notes

- Data has already been cleaned and split into `data/split/`. Do not recreate or modify the data files.
- See `src/config.py` for the full `Config` dataclass with all available parameters and defaults.
- Confirm with the user before starting long-running operations.

---

# Step 0: Data Exploration & Column Mapping

Before training anything, understand the dataset and configure the framework to use it.

## Inspect the Data

Load the training data from `data/split/` and report: number of samples, feature dimensions, time resolution (confirm 5-minute / 300-second), and any missing/NaN values. Understand what each column represents physically.

## Classify Columns

Assign every data column to one of Y, U, or D:

| Category | What it is | Examples |
|----------|-----------|----------|
| **Y** (states) | Variables the model predicts forward in time. These are the system's state. | Zone air temperatures, humidity, CO2 concentration |
| **U** (controls) | Controllable inputs that an operator or BMS can change. | Supply air temperature, airflow rate, cooling/heating setpoint, valve position, fan speed |
| **D** (disturbances) | Uncontrollable external inputs that affect the system. | Outdoor temperature, solar irradiance, occupancy, wind speed, internal heat gains |

**Guidelines:**
- If unsure whether something is U or D, ask: "Can the building operator change this?" If yes, it's U.
- Exclude columns that are metadata, identifiers, or derived from other columns (e.g., timestamps, indices).
- Constant or near-constant columns add no information — consider excluding them.
- A column can only be in ONE category. The framework raises `ValueError` on overlaps.

Confirm the classification with the user before proceeding.

## Edit `data_mapping()` in `src/user_defined.py`

The function uses two matching modes:

**Exact matching** (preferred for clarity):
```python
Y_EXACT = ["Zone 1 Air Temperature [C]", "Zone 2 Air Temperature [C]"]
U_EXACT = ["Supply Air Flow [m3/s]"]
D_EXACT = ["Outdoor Temperature [C]", "Solar Radiation [W/m2]"]
```

**Partial matching** (when column names follow a consistent pattern):
```python
Y_PARTIAL = []
U_PARTIAL = ["Supply Air", "Setpoint"]  # matches any column containing these substrings
D_PARTIAL = []
```

Prefer EXACT matches unless partial matching is unambiguous. Partial matches are convenient for systems with many similarly-named columns (e.g., 10 VAV boxes), but can cause accidental overlap if patterns are too broad.

**Also set:**
- `EXCLUDE_EXACT` / `EXCLUDE_PARTIAL` for columns to ignore entirely.

**Important:** The `sec_elapsed` column is created automatically by the framework from timestamps. Do not include it in Y/U/D — it is always mapped to `t`.

---

# Iterative Training Loop

After Step 0 is complete, enter an iterative refinement loop. Each iteration trains a model, evaluates it, diagnoses issues, and proposes targeted changes for the next round. Start simple and add complexity only when justified.

## Iteration 1: Baseline (No Physics)

Create a YAML config in `src/configs/` for baseline training with no physics losses. Set an informative `study_name` in the config.

**Omitted fields** inherit from the `Config` dataclass defaults. See `src/config.py` for the full class and default values.

**Input Window Constraint:** The model may use 1 to 12 timesteps (up to 1 hour at 5-minute resolution) as its input window. This is a hard upper bound. A shorter window is perfectly acceptable and encouraged if it performs well — a leaner, faster model is preferred over a marginally better slow one.

**Test Prediction Steps:** `test_pred_steps` should be fixed at 287 (approximately 1 day at 300s resolution) unless the user requests otherwise.

**What to check after training:**
1. **No runtime errors.** Common issues:
   - `KeyError` on column names → typo in `data_mapping()`
   - Shape mismatches → wrong data paths or inconsistent splits
   - `ZeroDivisionError` in normalization → constant column (stretch=0)
2. **Training converges.** Dev loss should decrease over epochs.
3. **Plots look reasonable.** Check `src/plots/` for prediction traces. The model should track the general trend of Y columns even if imperfect.
4. **MLflow logging works.** Run `uv run mlflow ui` and check for your run.

If training fails or produces garbage, debug before proceeding. Common fixes:
- Reduce `pred_steps` (shorter rollout = easier to learn initially)
- Check that U and D columns actually carry useful signal (not all constant)
- Verify data is properly aligned across splits (same columns, same frequency)

## Iteration 2+: Diagnose, Change, Retrain

At each subsequent iteration, follow this cycle:

### 1. Diagnose

Before changing anything, write a short analysis of the current model's performance. Be specific and reference actual numbers. Consider:
- Is the model overfitting (low training loss but poor test rollout)?
- Is the model underfitting (high loss, poor expressivity)?
- Are errors accumulating during autoregressive rollout (early timesteps fine, late timesteps diverge)?
- Are certain Y outputs or time periods significantly worse than others — and if so, why?
- Does the model violate physical intuition (e.g., cooling makes zones warmer)?

### 2. Propose Changes

State clearly what you intend to change and why that change addresses the diagnosed issue. Changes may include (but are not limited to):

**Model & training adjustments:**
- Architecture type (`mlp`, `nssm`, `lssm`, `gru`), depth, width
- Input window length (within the 12-timestep cap)
- `pred_steps` (training rollout horizon), learning rate, epochs, batch size
- `residual` mode

**Adding physics (when justified by diagnosis):**
- Gradient sign constraints (`grad_gain`, gradient mapping JSON)
- Synthetic scenario penalties (`synth_gain`, `synth_cfg()`)

**Physics configuration details — when you decide to add them:**

To add **gradient sign constraints**, create a JSON mapping in `data/grad/`. Format: `{output_col: {input_col: +1 or -1}}` where `+1` means increasing the input should increase the output, `-1` means the opposite. Common building thermal relationships:

| Input | Effect on zone temperature | Sign |
|-------|--------------------------|------|
| Outdoor temperature | Higher outdoor = warmer zone | +1 |
| Solar irradiance | More sun = warmer zone | +1 |
| Other zone temperature | Thermal coupling | +1 |
| Cooling setpoint | Higher setpoint = less cooling = warmer | +1 |
| Supply air temp (cooling) | Higher supply temp = warmer zone | +1 |
| Supply air flow (cooling) | More airflow = more cooling = colder | -1 |
| Heating setpoint | Higher setpoint = more heating = warmer | +1 |
| Valve position (cooling) | More open = more cooling = colder | -1 |
| Valve position (heating) | More open = more heating = warmer | +1 |

Only constrain relationships you are confident about. Wrong constraints hurt quality. You can generate combinatorial mapping files via `make_all_gradient_mappings()` in `user_defined.py`, or write the JSON directly.

To add **synthetic scenario penalties**, edit `synth_cfg()` in `user_defined.py`. Define physical operating values for every U and D column (in physical units, not normalized). The framework tests three scenarios:
1. **Equilibrium**: All zones at same temp, HVAC off, outdoor = indoor. Zones should stay constant.
2. **Step response**: HVAC turns on from equilibrium. Zones should move in the correct direction.
3. **Free decay**: Outdoor temp jumps while HVAC is off. Zones should drift toward outdoor temp.

For each U column, define `"off"` (inactive state) and optionally `"active_cooling"` and/or `"active_heating"`. For each D column, define `"off"` (neutral state). Mark exactly one D column with `"outdoor_temp": True` for free decay. Use `norm_stats[col]["min"]` and `norm_stats[col]["max"]` to reference observed data ranges. Every U and D column must appear in the returned dict.

**Prefer simpler, faster changes first.** Do not jump to a massive model if a learning rate adjustment might suffice. If a change significantly increases training time, acknowledge the trade-off.

### 3. Retrain and Evaluate

Create or update the YAML config, train, and evaluate. Compare against previous iterations on the same metrics. Check prediction plots and (if physics is enabled) synthetic scenario plots in `src/plots/synthetic/`.

### 4. Repeat

Continue iterating for a maximum of 5 iterations or until performance is satisfactory or further improvements yield diminishing returns. If you've tried several meaningful changes and performance has plateaued, it's reasonable to stop and report.

---

# Final Report

When you are done iterating, create `final_report.md` documenting:

- **Dataset overview**: What data was used, column classifications, any notable characteristics.
- **Iteration log**: For each iteration, describe:
  - What the model configuration was
  - What metrics it achieved (training loss, dev loss, test rollout performance)
  - What you diagnosed as the main issue
  - What you changed for the next iteration and why
- **Final model**: Which iteration produced the best model, its configuration, and its performance.
- **What worked and what didn't**: Which changes had the biggest impact? Which didn't help? What would you try next if given more time?

The report should be readable by someone who wasn't watching the process — they should understand the full arc from baseline to final model.
