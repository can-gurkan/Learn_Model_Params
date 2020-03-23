# Learning Model Parameters
Uses the REINFORCE policy gradient algorithm for learning the parameters of Agent-Based Models (ABMs) to produce realistic outputs 

## Problem Statement
In this work, we propose a reinforcement learning-based method for automatically calibrating the parameters of any (non-differentiable) simulator, thereby controlling the output of such a simulator in order to minimize the difference between the generated output and a given data set, such as real world data. We specifically aim to apply it to tuning the parameters of Agent-Based Models so that they would produce outputs similar to real world data.

### Input/Output
The input is the number of model parameters that should be calibrated. The output is the data the simulation model produces with updated model parameters.

## Run abm_param_learn.py
Install PyTorch here https://pytorch.org/get-started/locally/
Install NetLogo here https://ccl.northwestern.edu/netlogo/
and configure the path in abm_model.py to match the path of NetLogo in the directory you installed it in.

## Run abm_param_learn_pynetlogo.py
In addition to installing PyTorch and NetLogo (see above) this alternate version requires installing pyNetLogo here https://pynetlogo.readthedocs.io/en/latest/
The benefit is that this version runs faster than the abm_param_learn.py
