# Powerful Attacks Produce Strong Defenses: Two-Player Reinforcement Learning for Prompt Injection Defense

## Attribution
This repo is adapted from the [RL-Hammer](https://github.com/facebookresearch/rl-injector) repo and uses the [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent/tree/main/data) dataset.

## Data
Visit the [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent/tree/main/data) repo, dowload `test_cases_dh_base.json`,
and move to `data/InjecAgent/raw', and then run `python data/InjecAgent/split_dataset.py`.

## Launch scripts
The `launch_scripts/` directory contains scripts to launch the attack and defense agents.

```sh
# run adversarial training
./launch_scripts/defender.sh

# evaluate results
./launch_scripts/eval.sh
```

The `analysis/` directory contains scripts to analyze and plot the results.
