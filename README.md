# Comparison of Various Meta-Learning Paradigms in Few-Shot Preference-Based Reinforcement Learning

This project was undertaken as part of the course EECS 545: Machine Learning, at the University of Michigan, Ann Arbor, under the guidance of Professor Honglak Lee. We take a look at [Few-Shot Preference Learning for Human-in-the-Loop RL](https://arxiv.org/abs/2212.03363) which was published by Joey Hejna and Dorsa Sadigh in 2022. Although the code for their implementation has been published on [GitHub](https://github.com/jhejna/few-shot-preference-rl), we faced a bunch of issues when trying to run their code. As a consequence, we wrote our own models from scratch, and used their code only for data generation.

We experiment with three learning algorithms
1. Model-Agnostic Meta-Learning (MAML)
2. Iterated Model-Agnostic Meta-Learning (iMAML)
3. REPTILE

We also experiment with prior policies by augmenting our dataset with data generated from a prior expert policy, in the hopes of utilizing both human feedback and expert knowledge to quickly adapt our reward function to an unseen task.

The `datasets` directory contains all the data we have used. `mw` contains data for the 10 tasks used for pre-training. `mw_valid` contains data for the 4 tasks we adapt our reward functions to. `mw_valid_policy_v1` contains data for these same 4 tasks generated via expert policies. 

The `models` directory contains three files for our saved models (one for MAML, one for iMAML, and one for REPTILE). The `scripts` directory contains our code for all models. The `train_MAML.py`, `train_iMAML.py`, and `train_Reptile.py` contain code for the three corresponding models, with the corresponding `.ipynb` notebooks containing their train and adaptation runs. `pebble.py` contains the code for adapting the saved network to a new unseen task. The `Plots.ipynb` notebook generates plots from our saved results.