import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import optuna as op

####################################################################

#Plotting data
cases_df = pd.read_excel('cum_severe_wave1.xlsx', index_col = 'date', parse_dates = True)

fig = plt.figure(figsize=(10,8))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

daily_cases_df = cases_df.diff(periods = -1)

ax1.plot(cases_df.index, cases_df.values, linewidth=0.75)
ax1.grid()
ax1.set_title('Cumulative hospitalisations')
ax2.plot(daily_cases_df.index, daily_cases_df.values, linewidth=0.75)
ax2.set_title('New daily hospitalisations')
ax2.grid()

fig.autofmt_xdate()
plt.show()

######################################################################

#Parameter analysis plots

name      = 'covasim_uk_calibration_wave1_deaths_100trials'
storage   = f'sqlite:///{name}.db'
study = op.load_study(study_name=name, storage=storage)
output = study.best_params

study_data = pd.read_json('calibrated_parameters_UK.json')
pars_df = pd.json_normalize(study_data['pars'])
pars_df = pars_df.assign(trials = study_data['index'].values)
print(pars_df.head())
beta_df = pars_df[['trials','beta']]

print("Value of Beta given by Optuna = "+str(output['beta']))

edf = op.visualization.plot_contour(study, params=["beta", "pop_infected"])
edf.add_trace(go.Scatter(x=[output['beta']], y=[output['pop_infected']],
                    mode='markers',
                    marker_size = 20,
                    name='Optimal value',
                    marker_color='rgba(152, 0, 0, .8)'))

edf.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01))

edf.show()

edf.write_image("param_search_wave1_100trials.pdf")


#2D parameter printing
