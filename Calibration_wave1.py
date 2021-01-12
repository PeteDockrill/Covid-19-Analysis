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

def create_sim(x, start_date, end_date):

    beta = x[0]
    pop_infected = x[1]

    start_day = '2020-01-21' #Start of the simulation
    end_day   = '2020-06-29' #End of the simulation
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

    #h = household, s = schools, w = workplaces and c = community 
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



def objective(x):
    ''' Define the objective function we are trying to minimize '''

    # Create and run the sim
    sim = create_sim(x)
    sim.run(verbose = 0.1)
    fit = sim.compute_fit()

    return fit.mismatch


def get_bounds():
    ''' Set parameter starting points and bounds '''
    pdict = sc.objdict(
        beta         = dict(best=0.00522, lb=0.003, ub=0.008),
        pop_infected = dict(best=4500,  lb=1000,   ub=10000),
    )

    # Convert from dicts to arrays
    pars = sc.objdict()
    for key in ['best', 'lb', 'ub']:
        pars[key] = np.array([v[key] for v in pdict.values()])

    return pars, pdict.keys()


#%% Calibration

name      = 'covasim_uk_calibration_jan_june_severe_700_trials_pop_infect'
storage   = f'sqlite:///{name}.db'
n_trials  = 175 #originally 100
n_workers = 4

pars, pkeys = get_bounds() # Get parameter guesses


def op_objective(trial):

    pars, pkeys = get_bounds() # Get parameter guesses
    x = np.zeros(len(pkeys))
    for k,key in enumerate(pkeys):
        x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

    return objective(x)


def worker():
    study = op.load_study(storage=storage, study_name=name)
    return study.optimize(op_objective, n_trials=n_trials)


def run_workers():
    return sc.parallelize(worker, n_workers)


def make_study():
    try: op.delete_study(storage=storage, study_name=name)
    except: pass
    return op.create_study(storage=storage, study_name=name)


def calibrate():
    ''' Perform the calibration '''
    make_study()
    run_workers()
    study = op.load_study(storage=storage, study_name=name)
    output = study.best_params
    return output, study


def savejson(study):
    dbname = 'covasim_uk_calibration_jan_june_severe_700trials_pop_infect'

    sc.heading('Making results structure...')
    results = []
    failed_trials = []
    for trial in study.trials:
        data = {'index':trial.number, 'mismatch': trial.value}
        for key,val in trial.params.items():
            data[key] = val
        if data['mismatch'] is None:
            failed_trials.append(data['index'])
        else:
            results.append(data)
    print(f'Processed {len(study.trials)} trials; {len(failed_trials)} failed')

    sc.heading('Making data structure...')
    keys = ['index', 'mismatch'] + pkeys
    data = sc.objdict().make(keys=keys, vals=[])
    for i,r in enumerate(results):
        for key in keys:
            data[key].append(r[key])
    df = pd.DataFrame.from_dict(data)

    order = np.argsort(df['mismatch'])
    json = []
    for o in order:
        row = df.iloc[o,:].to_dict()
        rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
        for key,val in row.items():
            rowdict['pars'][key] = val
        json.append(rowdict)
    sc.savejson(f'{dbname}.json', json, indent=2)

    return

###############################################################################

if __name__ == '__main__':

    #do_save = True

    # Plot initial
    print('Running initial...')
    pars, pkeys = get_bounds() # Get parameter guesses
    init_sim = create_sim(pars.best)
    init_sim.run(verbose = 0.1)
    objective(pars.best)
    pl.pause(1.0) # Ensure it has time to render

    # Calibrate
    print('Starting calibration for {state}...')
    T = sc.tic() #sciris timing operations
    pars_calib, study = calibrate()
    sc.toc(T) #Another sciris timing operation

    if do_save:
        savejson(study)

    ###########################################################################
    #If calibration has already been run:

    #Load optuna study
    #name      = 'covasim_uk_calibration_jan_june_severe_700_trials_pop_infect'
    #storage   = f'sqlite:///{name}.db'
    #study = op.load_study(study_name=name, storage=storage)
    #pars_calib = study.best_params

    #start_date = '2020-01-21' #Start of the simulation
    #end_date   = '2020-06-29' #End of the simulation
    #Create simulations
    #print("Running simulation with Beta = "+str(pars_calib['beta'])+'.')
    #sim = create_sim([pars_calib['beta'], pars_calib['pop_infected']], start_date, end_date)

    ###########################################################################
    #Parameter space plots with Optuna and Plotly

    #print("Value of Beta given by Optuna = "+str(pars_calib['beta']))

    #edf = op.visualization.plot_contour(study, params=["beta",'pop_infected'])
    #edf.add_trace(go.Scatter(x=[pars_calib['beta']], y=[pars_calib['pop_infected']],
    #                    mode='markers',
    #                    marker_size = 15,
    #                    name='Optimal value (Beta = '+str(round(pars_calib['beta'], 6))+' pop_infected = '+str( round(pars_calib['pop_infected'],0))+')',
    #                    marker_color='rgba(152, 0, 0, .8)'))

    #edf.update_layout(legend=dict(
    #    yanchor="top",
    #    y=0.99,
    #    xanchor="left",
    #    x=0.01))

    #edf.show()

    #edf.write_image("param_search_wave1_severe_700trials_pop_infect.pdf")

    ############################################################################
    #Evaluate the quality of the calibration

    sim = create_sim([pars_calib['beta'], pars_calib['pop_infected']])
    sim.run()
    fit = sim.compute_fit()
    error_plot = fit.plot(do_show = False)
    error_plot[0].savefig('error_plot_severe_700_pop_infect.pdf')

print('Done.')
