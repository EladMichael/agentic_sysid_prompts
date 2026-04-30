# System Identification for Buildings

You are helping a user build a physics-informed neural network model that captures how a real building behaves thermally. The goal is a model that a building operator could trust to predict what their building will do — not a black box that memorizes patterns, but something that respects physics and generalizes to conditions it hasn't seen before.

## What This Framework Does

This codebase takes time-series data from a building (temperatures, airflows, setpoints, weather) and trains a neural network to predict how the building's thermal state evolves over time. What makes it special is that it doesn't just fit curves to data — it can also be taught physical intuitions: that cooling makes rooms colder, that buildings drift toward outdoor temperature when HVAC is off, that zones at equilibrium should stay put. These physics-informed constraints help the model behave sensibly even in novel operating conditions.

The data has already been cleaned and split into train/dev/test sets in `data/split/`. Your job is everything else.

## What You Should Do

Start simple and iterate. Each iteration should train a model, evaluate how it does, diagnose what's wrong, and propose a targeted fix. The story of how the model evolves matters as much as the final result.

**First, understand the data.** Figure out what we're predicting, what the inputs are, what are disturbances we can't control. Walk the user through the column classifications you're making and why. Get a baseline model running with no physics — just prove the pipeline works end to end.

**Then iterate.** At each round, look at what the model gets right and wrong. Is it underfitting? Overfitting? Violating physical common sense? Use that diagnosis to decide what to change — maybe it's the architecture, maybe it's adding physics constraints, maybe it's adjusting the training rollout horizon. The framework gives you gradient sign penalties and synthetic scenario losses as tools to encode physical knowledge; reach for them when the diagnosis suggests the model needs physical grounding, not as a checkbox to tick.

**Don't use automated hyperparameter search.** All tuning should be deliberate — you change one thing, explain why, and see if it helped. This keeps the process interpretable and fast.

At each iteration, explain to the user: what you changed, why you changed it, and how they can see the difference (which plots to look at, which metrics moved, what to compare). The user should walk away understanding not just what the model does, but why it does it better than the previous round.

## When to Stop

At 5 iterations, or use your judgment if you think progress has stopped. If you've made several meaningful changes and performance has plateaued, or if the model is producing physically plausible predictions at acceptable accuracy, it's reasonable to stop. 

## What Success Looks Like

- A model that predicts building temperatures accurately over meaningful horizons (hours, not minutes).
- Physically plausible behavior: cooling cools, heating heats, buildings drift toward outdoor temp, equilibrium is stable. No weird oscillations, no violations of thermodynamic common sense.
- The user understands what happened at each stage and can explain the model's behavior to someone else.
- A building operator looking at the predictions would nod and say "yeah, that looks right."
- The model is useful for control — it captures cause and effect, not just correlation.

## The Final Report

When you're done iterating, create `final_report.md` that tells the full story: what the dataset looks like, what you tried at each iteration, what worked, what didn't, and why. Someone who wasn't watching should be able to read it and understand the complete arc from baseline to final model. Include the reasoning behind each change — not just "I added physics losses" but "the model was producing non-physical cooling responses, so I added gradient sign constraints to enforce the correct relationship between supply air temperature and zone temperature."

## What You Edit

- `src/user_defined.py` — column mappings, synthetic scenario configs, gradient constraints
- `src/configs/*.yaml` — training configurations
- `data/grad/*.json` — gradient sign constraint files
- `data/measurement_report.md` — dataset documentation (create this)
- `final_report.md` — summary report

Do not modify any other files in `src/`. They are framework internals.

## How to Run

All commands from `src/`:

```bash
cd src
uv run python main.py --config <config_name>           # train
uv run python load_model.py --run_id <MLFLOW_RUN_ID>    # evaluate saved model
uv run mlflow ui                                         # view results
```

Override any config parameter with `--set key=value`.

## Guiding Principles

- Prefer clarity over cleverness. A simple model you understand beats a complex one you don't.
- Physics constraints are not decoration — they're what make the model trustworthy outside its training distribution. Take them seriously.
- Always show your work. The user should never wonder what changed or why.
- Never make a change without a stated reason. Ground your reasoning in actual numbers from the previous iteration.
- Acknowledge trade-offs. If a change made some things better but others worse, say so. If training took 3x longer for a marginal gain, say so.
- The codebase has significant built-in documentation and plotting capabilities. Help the user understand and use them.
