import sciris as sc
import covasim as cv
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import optuna as op
import Dec_2020_Calibration_wave1 as cb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

print("#####################################################")
print("               Beginning Simulations                 ")
print("#####################################################")

def create_sim(x, start, end):

    beta = x[0]
    pop_infected = x[1]

    start_day = start #Start of the simulation
    end_day   = end #End of the simulation
    data_path = 'cum_severe_wave1.xlsx'

    # Set the parameters
    total_pop    = 67.86e6 # UK population size
    pop_size     = 100e3 # Actual simulated population
    pop_scale    = int(total_pop/pop_size)
    pop_type     = 'hybrid'
    asymp_factor = 2
    contacts     = {'h':3.0, 's':20, 'w':20, 'c':20}

    pars = sc.objdict(
        pop_size     = pop_size,
        pop_infected = pop_infected,
        pop_scale    = pop_scale,
        pop_type     = pop_type,
        start_day    = start_day,
        end_day      = end_day,
        beta         = beta,
        asymp_factor = asymp_factor,
        contacts     = contacts,
        rescale      = True,
        verbose      = 0.1,
    )

    # Create the baseline simulation
    sim = cv.Sim(pars=pars, datafile=data_path, location='uk')

    #N.B - Interventions taken from UK_Masks_TTI_19Oct.py
    tc_day = sim.day('2020-03-16') #intervention of some testing (tc) starts on 16th March and we run until 1st April when it increases
    te_day = sim.day('2020-04-01') #intervention of some testing (te) starts on 1st April and we run until 1st May when it increases
    tt_day = sim.day('2020-05-01') #intervention of increased testing (tt) starts on 1st May
    tti_day= sim.day('2020-06-01') #intervention of tracing and enhanced testing (tti) starts on 1st June
    ti_day = sim.day('2021-12-20') #schools interventions end date in December 2021
    tti_day_july= sim.day('2020-07-01') #intervention of tracing and enhanced testing (tti) at different levels starts on 1st July
    tti_day_august= sim.day('2020-08-01') #intervention of tracing and enhanced testing (tti) at different levels starts on 1st August
    tti_day_sep= sim.day('2020-09-01') #intervention of tracing and enhanced testing (tti) at different levels starts on 1st September
    tti_day_oct= sim.day('2020-10-01') #intervention of tracing and enhanced testing (tti) at different levels starts on 1st October

    #change parameters here for different schools opening strategies with society opening
    beta_days = ['2020-02-14', '2020-03-16', '2020-03-23', '2020-04-30', '2020-05-15', '2020-06-01', '2020-06-15', '2020-07-22', '2020-08-01', '2020-09-02', '2020-10-01', '2020-10-16', '2020-10-28', '2020-11-01', '2020-12-23', '2021-01-03', '2021-01-20', '2021-02-17', ti_day]

    h_beta_changes = [1.00, 1.00, 1.29, 1.29, 1.29, 1.00, 1.00, 1.29, 1.29, 1.00, 1.00, 1.00, 1.29, 1.00, 1.29, 1.00, 1.00, 1.29, 1.00]
    s_beta_changes = [1.00, 0.90, 0.02, 0.02, 0.02, 0.23, 0.38, 0.00, 0.00, 0.63, 0.63, 0.63, 0.00, 0.63, 0.00, 0.63, 0.63, 0.00, 0.63]
    w_beta_changes = [0.90, 0.80, 0.20, 0.20, 0.20, 0.40, 0.40, 0.60, 0.60, 0.60, 0.60, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    c_beta_changes = [0.90, 0.80, 0.20, 0.20, 0.20, 0.40, 0.50, 0.60, 0.60, 0.70, 0.70, 0.70, 0.60, 0.60, 0.50, 0.60, 0.60, 0.50, 0.50]

    # Define the beta changes
    h_beta = cv.change_beta(days=beta_days, changes=h_beta_changes, layers='h')
    s_beta = cv.change_beta(days=beta_days, changes=s_beta_changes, layers='s')
    w_beta = cv.change_beta(days=beta_days, changes=w_beta_changes, layers='w')
    c_beta = cv.change_beta(days=beta_days, changes=c_beta_changes, layers='c')

    #next line to save the intervention
    interventions = [h_beta, w_beta, s_beta, c_beta]

    # Tracing and enhanced testing strategy of symptimatics from 1st June
    s_prob_march = 0.009
    s_prob_april = 0.013
    s_prob_may   = 0.027
    s_prob_june = 0.02769
    s_prob_july = 0.02769
    s_prob_august = 0.02769
    s_prob_sept = 0.05769
    s_prob_oct = 0.08769
    t_delay       = 1.0

    iso_vals = [{k:0.1 for k in 'hswc'}] #i.e 90% adherence to isolation
    iso_vals1 = [{k:0.8 for k in 'hswc'}] #i.e 20% adherence to isolation
    iso_vals2 = [{k:0.8 for k in 'hswc'}] #i.e 20% adherence to isolation

    #tracing level at 42.35% in June; 47.22% in July, 44.4% in August and 49.6% in September (until 16th Sep)
    t_eff_june   = 0.42
    t_eff_july   = 0.47
    t_eff_august = 0.44
    t_eff_sep    = 0.50
    t_eff_oct    = 0.50
    t_probs_june = {k:t_eff_june for k in 'hwsc'}
    t_probs_july = {k:t_eff_july for k in 'hwsc'}
    t_probs_august = {k:t_eff_august for k in 'hwsc'}
    t_probs_sep = {k:t_eff_sep for k in 'hwsc'}
    t_probs_oct = {k:t_eff_oct for k in 'hwsc'}
    trace_d_1      = {'h':0, 's':1, 'w':1, 'c':2}

    #testing and isolation intervention
    interventions += [
        cv.test_prob(symp_prob=0.009, asymp_prob=0.0, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=tc_day, end_day=te_day-1, test_delay=t_delay, test_sensitivity=0.97),
        cv.test_prob(symp_prob=s_prob_april, asymp_prob=0.0, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=te_day, end_day=tt_day-1, test_delay=t_delay, test_sensitivity=0.97),
        cv.test_prob(symp_prob=s_prob_may, asymp_prob=0.0075, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=tt_day, end_day=tti_day-1, test_delay=t_delay, test_sensitivity=0.97),
        cv.test_prob(symp_prob=s_prob_june, asymp_prob=0.0175, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=tti_day, end_day=tti_day_july-1, test_delay=t_delay, test_sensitivity=0.97),
        cv.test_prob(symp_prob=s_prob_july, asymp_prob=0.0175, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=tti_day_july, end_day=tti_day_august-1, test_delay=t_delay, test_sensitivity=0.97),
        cv.test_prob(symp_prob=s_prob_august, asymp_prob=0.0175, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=tti_day_august, end_day=tti_day_sep-1, test_delay=t_delay,test_sensitivity=0.97),
        cv.test_prob(symp_prob=s_prob_sept, asymp_prob=0.0175, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=tti_day_sep, end_day=tti_day_oct-1, test_delay=t_delay, test_sensitivity=0.97),
        cv.test_prob(symp_prob=s_prob_oct, asymp_prob=0.0175, symp_quar_prob=0.0, asymp_quar_prob=0.0, start_day=tti_day_oct, test_delay=t_delay, test_sensitivity=0.97),
        cv.dynamic_pars({'iso_factor': {'days': te_day, 'vals': iso_vals}}),
        cv.contact_tracing(trace_probs=t_probs_june, trace_time=trace_d_1, start_day=tti_day, end_day=tti_day_july-1),
        cv.contact_tracing(trace_probs=t_probs_july, trace_time=trace_d_1, start_day=tti_day_july, end_day=tti_day_august-1),
        cv.contact_tracing(trace_probs=t_probs_august, trace_time=trace_d_1, start_day=tti_day_august, end_day=tti_day_sep-1),
        cv.contact_tracing(trace_probs=t_probs_sep, trace_time=trace_d_1, start_day=tti_day_sep, end_day=tti_day_oct-1),
        cv.contact_tracing(trace_probs=t_probs_sep, trace_time=trace_d_1, start_day=tti_day_oct),
        cv.dynamic_pars({'iso_factor': {'days': tti_day, 'vals': iso_vals}}),
        cv.dynamic_pars({'iso_factor': {'days': tti_day_august, 'vals': iso_vals1}}),
        cv.dynamic_pars({'iso_factor': {'days': tti_day_sep, 'vals': iso_vals2}}),
    ]


    sim.update_pars(interventions=interventions)
    for intervention in sim['interventions']:
        intervention.do_plot = False

    return sim

if __name__ == '__main__':

    to_plot = ['new_infections']

    # Save the key figures
    plot_customizations = dict(
        interval   = 90, # Number of days between tick marks
        dateformat = '%m/%Y', # Date format for ticks
        fig_args   = {'figsize':(14,8)}, # Size of the figure (x and y)
        axis_args  = {'left':0.15}, # Space on left side of plot

        )

    #Load optuna study
    name      = 'covasim_uk_calibration_jan_june_severe_100trials'
    storage   = f'sqlite:///{name}.db'
    study = op.load_study(study_name=name, storage=storage)
    output = study.best_params

    start_date = '2020-01-21'
    end_date = '2020-07-29'
    #Create simulations
    print("Running simulation with Beta = "+str(output['beta'])+'.')
    sim = create_sim([output['beta'], output['pop_infected']], start_date, end_date)
    msim = cv.MultiSim(sim, n_runs=10)
    msim.run()

    # Plot result
    print('Plotting result...')
    msim.reduce()
    msim_plot = msim.plot(to_plot = to_plot, fig_args={'figsize':(10,8), 'linewidth':0.75}, alpha_range = [0.1, 1.0])
    msim.summarize()

    #msim_plot_deaths = msim.plot_result('cum_deaths', **plot_customizations)
    #pl.title('')
    #msim_plot_2.savefig('Deaths_100_trials.pdf')

    msim_plot_hospitalisations = msim.plot_result('cum_severe', **plot_customizations)
    pl.title('')
    msim_plot_hospitalisations.savefig('Hospital_100_trials.pdf')

    ####################################################################

    #Plotting data
    cases_df = pd.read_excel('cum_severe_wave1.xlsx', index_col = 'date', parse_dates = True)

    fig = plt.figure(figsize=(10,8))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    daily_cases_df = cases_df.diff(periods = -1)

    ax1.plot(cases_df.index, cases_df.values, linewidth=0.75)
    ax1.grid()
    ax1.set_title('Cumulative diagnoses')
    ax2.plot(daily_cases_df.index, daily_cases_df.values, linewidth=0.75)
    ax2.set_title('New daily diagnoses')
    ax2.grid()

    fig.autofmt_xdate()
    plt.gcf()
    plt.show()

    ######################################################################

    #Parameter space plots

    print("Value of Beta given by Optuna = "+str(output['beta']))

    edf = op.visualization.plot_contour(study, params=["beta", "pop_infected"])
    edf.add_trace(go.Scatter(x=[output['beta']], y=[output['pop_infected']],
                        mode='markers',
                        marker_size = 15,
                        name='Optimal value (Beta = '+str(round(output['beta'], 6))+' pop_infected = '+str( round(output['pop_infected'],0))+')',
                        marker_color='rgba(152, 0, 0, .8)'))

    edf.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01))

    #edf.show()

    edf.write_image("param_search_wave1_severe_100trials.pdf")
