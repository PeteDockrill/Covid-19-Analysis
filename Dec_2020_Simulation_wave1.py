import sciris as sc
import covasim as cv
import pylab as pl
import numpy as np
import matplotlib as mplt
import optuna as op
import Dec_2020_Calibration_wave1 as cb

print("#####################################################")
print("               Beginning Simulations                  ")
print("#####################################################")

if __name__ == '__main__':
    #freeze_support()

    to_plot = ['new_infections']

    #Load optuna study
    name      = 'covasim_uk_calibration_wave1_deaths_300trials'
    storage   = f'sqlite:///{name}.db'
    study = op.load_study(study_name=name, storage=storage)
    output = study.best_params
    #Create simulations
    sim = cb.create_sim([output['beta'], output['pop_infected']])
    msim = cv.MultiSim(sim, n_runs=10) # Create the multisim
    msim.run()

    # Plot result
    print('Plotting result...')
    msim.reduce()
    msim.plot(to_plot = to_plot, fig_args={'figsize':(10,8), 'linewidth':0.75}, alpha_range = [0.1, 1.0])
    msim.summarize()
