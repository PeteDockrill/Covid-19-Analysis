import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cases_df = pd.read_excel('cum_severe_wave1.xlsx', index_col = 'date', parse_dates = True)

#Append zero to the front
daily_cases_df = cases_df.diff(periods = -1)
#daily_cases_df = daily_cases_df[daily_cases_df['cum_deaths']>0]

fig = plt.figure(figsize=(10,8))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(cases_df.index, cases_df.values, linewidth=0.75)
ax1.grid()
ax1.set_title('Cumulative hospitalisations')
ax2.plot(daily_cases_df.index, daily_cases_df.values, linewidth=0.75)
ax2.set_title('New daily hospitalisations')
ax2.grid()

fig.autofmt_xdate()
plt.show()
