import sciris as sc
import covasim as cv
import pylab as pl
import numpy as np
import matplotlib as mplt
import Dec_2020_Calibration_wave1 as cb

print("#####################################################")
print("               Begnning Simulations                  ")
print("#####################################################")
#Load optuna study
name      = 'covasim_uk_calibration'
storage   = f'sqlite:///{name}.db'
study = optuna.create_study(study_name='Optimal parameters', storage=storage, load_if_exists=True)

#Create simulations
sim = cb.create_sim(study.best_params)
msim = cv.MultiSim(sim, n_runs=10) # Create the multisim
msim.run()

# Plot result
print('Plotting result...')
msim.plot(to_plot = to_plot, fig_args={'figsize':(7,3.5), 'linewidth':0.75}, alpha_range = [0.1, 1.0])
msim.summarize()
