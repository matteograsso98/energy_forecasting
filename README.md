This repository contain two script to forecast energy power for photovoltaic plants (also usable for wind turbines). 

- 'EnergyForecasterNN.ipynb' approaches forecasting through a fully-connected neural network (FCNN) -
  - The notebook is based on 'CosmoPower' notebook by A.S.Mancini, which is an emulator used to speed up bayesian inference in cosmology
  - Shuffling the data set is the core question of the notebook, as Shuffle=False is a commonly adopted setup for time-series forecasting (for several reasons), but it actually depends on how we think about the problem.
- 'random_forest_pv_forecasting.ipynb' approaches forecasting through a Random Forest-based implementation. 
