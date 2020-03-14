# Learning Model Parameters
Uses the REINFORCE policy gradient algorithm for learning the parameters of Agent-Based Models (ABMs) to produce realistic outputs 

## Problem Statement
In this work, we propose a reinforcement learning-based method for automatically calibrating the parameters of any (non-differentiable) simulator, thereby controlling the output of such a simulator in order to minimize the difference between the generated output and a given data set, such as real world data. We specifically aim to apply it to tuning the parameters of Agent-Based Models so that they would produce outputs similar to real world data.

### Input/Output
The input is the number of model parameters that should be calibrated. The output is the data the simulation model produces with updated model parameters.
