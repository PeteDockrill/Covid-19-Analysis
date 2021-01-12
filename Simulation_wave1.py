import sciris as sc
import covasim as cv
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import optuna as op
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import covasim.analysis as cva

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

def create_msim(sim, n_sims, added_noise):

    msim = cv.MultiSim(sim, n_runs = n_sims, noise = added_noise)
    msim.run(verbose = 0.1)
    red_msim = msim.reduce(quantiles=None, output=True)

    #Plot new infections
    plot_customizations = dict(
        interval   = 90, # Number of days between tick marks
        dateformat = '%m/%Y', # Date format for ticks
        fig_args   = {'figsize':(14,8)}, # Size of the figure (x and y)
        axis_args  = {'left':0.15}, # Space on left side of plot
    )

    print('Plotting result...')
    msim_plot = msim.plot(to_plot = ['new_infections'], fig_args={'figsize':(10,8), 'linewidth':0.75}, alpha_range = [0.1, 1.0])

    #Plot hospitalisations with data
    msim_plot_hospitalisations = msim.plot_result('cum_severe', **plot_customizations)
    pl.title('')
    if added_noise == 0.00:
        msim_plot_hospitalisations.savefig('median_'+str(n_sims)+'_sims_severe_noiseless.pdf')
    else:
        msim_plot_hospitalisations.savefig('median_'+str(n_sims)+'_sims_severe_noise_'+str(added_noise)+'.pdf')

    return msim, red_msim

def parameter_search(n_simulations, load_previous, prev_sim_list):

    if load_previous == True:
        sims_df, best_parameters = load_past_searches(prev_sim_list)
    else:
        print("Initialising empty dataframe")
        sims_df = pd.DataFrame(columns = ['beta','pop_infected', 'mismatch'])

    #Run simultions with varied parameters in local region of optimal parameters

    for i in range (1,n_simulations+1):
        if n_simulations == 0:
            print("Not running any new simulations")
            break
        else:
            print("Running parameter search...")
            beta_vary = 0.0000001*((2 * np.random.random()) -1)
            pop_infected_vary = 1*((2 * np.random.random()) -1)

            new_beta = output['beta'] + beta_vary
            new_pop_infected = 1371.0 #output['pop_infected'] # + pop_infected_vary
            new_params = [new_beta, new_pop_infected]

            new_sim = create_sim(new_params, start_date, end_date)
            print('Running simulation '+str(i)+" of "+str(n_simulations))
            new_sim.run(verbose = 0)
            new_sim_fit = new_sim.compute_fit()
            new_sim_mismatch = new_sim_fit.compute_mismatch()

            new_sim_df = pd.DataFrame(data = {'beta':[new_beta], 'pop_infected':[new_pop_infected], 'mismatch':[new_sim_mismatch]})
            sims_df = sims_df.append(new_sim_df, ignore_index = True)

    sorted_sims_df, best_parameters = sort_dataframe(sims_df)
    sorted_sims_df.to_csv('optimal_param_search_'+str(n_simulations)+'_sims.csv')

    return sorted_sims_df

def sort_dataframe(dataframe):
    "Sorts values of a dataframe according to increasing mismatch"

    print("Sorting data...")
    sorted_sims_df = dataframe.sort_values(by = ['mismatch'], ignore_index = True)
    best_parameters = [sorted_sims_df.beta[0],sorted_sims_df.pop_infected[0]]
    print("Best trial: beta = "+str(best_parameters[0])+", pop_infected = "+str(best_parameters[1])+" with a mismatch of "+str(sorted_sims_df.mismatch[0]))
    print("Best 10 simulations:")
    print(sorted_sims_df.head(10))

    return sorted_sims_df, best_parameters

def load_past_searches(runs_list):
    "Reloads data from previous searches that have been saved as .csv files"

    print("Loading data from previous runs...")
    past_search_df = pd.DataFrame(columns = ['beta','pop_infected', 'mismatch'])
    for i, number in enumerate(runs_list):
        df_i = pd.read_csv('optimal_param_search_'+str(number)+'_sims.csv')
        past_search_df = past_search_df.append(df_i, ignore_index = True)

    past_search_df, best_parameters = sort_dataframe(past_search_df)

    return past_search_df

if __name__ == '__main__':

    #Load optuna study
    name      = 'covasim_uk_calibration_jan_june_severe_700_trials_pop_infect'
    storage   = f'sqlite:///{name}.db'
    study = op.load_study(study_name=name, storage=storage)
    output = study.best_params

    start_date = '2020-01-21'
    end_date = '2020-07-29'

    #Create base simulation
    print("Optimal parameters from calibration: Beta = "+str(output['beta'])+', pop_infected = '+str(output['pop_infected'])+'.')
    # base_sim = create_sim([output['beta'], output['pop_infected']], start_date, end_date)

    ############################################################################
    #Simulaitons excluding noise
    #noise = 0.00

    #msim_50_sims, red_msim_50 = create_msim(base_sim, 50, noise)
    #msim_100_sims, red_msim_100 = create_msim(base_sim, 100, noise)
    #msim_300_sims, red_msim_300 = create_msim(base_sim, 300, noise)
    #msim_400_sims, red_msim_400 = create_msim(base_sim, 400, noise)

    #Median comparison
    #reduced_msims = [red_msim_50, red_msim_100, red_msim_300, red_msim_400]
    #reduced_msim_labels = ['50 sims', '100 sims', '300 sims', '400 sims']

    #median_msim = cv.MultiSim(reduced_msims)
    #mediam_msim.run()
    #median_plot = median_msim.plot(to_plot =['cum_severe'], plot_sims = True, labels = reduced_msim_labels)
    #median_plot.savefig("median_plot")

    ############################################################################
    #Optimal parameter space search
    df_200_sims = parameter_search(47, False, [])

    #print(sims_df.head())

    #parameter_search_msim = cv.MultiSim(sims)
    #parameter_search_msim

    ############################################################################
    #Plotting data

    #cases_df = pd.read_excel('cum_severe_wave1.xlsx', index_col = 'date', parse_dates = True)
    #daily_cases_df = cases_df.diff(periods = -1)

    #fig = plt.figure(figsize=(10,8))
    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)

    #ax1.plot(cases_df.index, cases_df.values, linewidth=0.75)
    #ax1.grid()
    #ax1.set_title('Cumulative diagnoses')
    #ax2.plot(daily_cases_df.index, daily_cases_df.values, linewidth=0.75)
    #ax2.set_title('New daily diagnoses')
    #ax2.grid()

    #fig.autofmt_xdate()
    #plt.gcf()
    #plt.show()

    ######################################################################
    #Median comparison
