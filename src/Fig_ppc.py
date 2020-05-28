#!/usr/bin/env python
# coding: utf-8
import sys
import os
from datetime import datetime, timedelta
import urllib

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.integrate import odeint
import scipy.signal
import pandas as pd
import seaborn as sns
sns.set_context('paper', font_scale=1.3)
red, blue, green = sns.color_palette('Set1', 3)
colors = {'red':red, 'blue':blue, 'green':green}

from click_spinner import spinner

from inference import get_last_NPI_date
from inference import get_first_NPI_date
from inference import params_bounds
from inference import get_model_class
from inference import find_start_day
from model.normal_prior_model import NormalPriorModel
from model.fixed_tau_model import FixedTauModel


def int_to_dt(t):
    return pd.to_datetime(start_date) + timedelta(days=t)

def date_to_int(x):
    dt = datetime.strptime(x + ' 2020', '%b %d %Y')
    td = dt - start_date
    return td.days

def date_to_date(x):
    dt = datetime.strptime(x + ' 2020', '%b %d %Y')
    return dt

def τ_to_string(τ, start_date):
    return (pd.to_datetime(start_date) + timedelta(days=τ)).strftime('%b %d')

def load_chain(job_id=None, fname=None, burn_fraction=0.6):
	with spinner():
		if fname is None:
			fname = os.path.join(output_folder, job_id, 'inference', '{}.npz'.format(country))
		inference_data = np.load(fname)
		chain = inference_data['chain']
		var_names = inference_data['var_names']
		nsteps, ndim, N, Td1, Td2, model_type = inference_data['params']
		X = inference_data['incidences']
		start_date = inference_data['start_date']
		logliks = inference_data['logliks']
		# print("Loaded {} with parameters:".format(fname))
		# print(var_names)
		nchains, nsteps, ndim = chain.shape
		nburn = int(nsteps * burn_fraction)
		chain = chain[:, nburn:, :]
		chain = chain.reshape((-1, ndim))
		logliks = logliks.reshape((nchains, nsteps))
		logliks = logliks[:, nburn:].ravel()
		return chain, logliks, Td1, Td2, model_type, X, start_date, N

def posterior_prediction(chain, model, nreps):
	θ = chain[np.random.choice(chain.shape[0], nreps)]
	return np.array([
		model.generate_daily_cases(θi) for θi in θ
	])


def load_data(country_name, up_to_date=None):
	if country_name=='Wuhan':
	    df = pd.read_csv('../data/Incidence.csv')
	    df['date'] = pd.to_datetime(df['Date'], dayfirst=True)
	    df['cases'] = df[country_name]
	    df = df[::-1] # TODO why?
	    N = pd.read_csv('../data/pop.csv', index_col='City').loc[country_name].values[0]
	else:
	    url = 'https://github.com/ImperialCollegeLondon/covid19model/raw/master/data/COVID-19-up-to-date.csv'
	    fname = '../data/COVID-19-up-to-date_master.csv'
	    if not os.path.exists(fname):
	        urllib.request.urlretrieve(url, fname)
	    df = pd.read_csv(fname, encoding='iso-8859-1')
	    df['date'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
	    df = df[df['countriesAndTerritories'] == country_name]
	    N = df.iloc[0]['popData2018']

	cases_and_dates = df.iloc[::-1][['cases','date']]
	if up_to_date:
	    cases_and_dates = cases_and_dates[cases_and_dates['date']<=up_to_date]
	start_date = find_start_day(cases_and_dates)
	X = cases_and_dates.loc[cases_and_dates['date'] >= start_date, 'cases'].values
	T = cases_and_dates.loc[cases_and_dates['date'] >= start_date, 'date']
	return X, T


if __name__ == '__main__':
	nreps = 1000
	date_threshold = datetime(2020, 3, 28)
	last_date = datetime(2020, 3, 28) + timedelta(7)

	output_folder = r'../output'
	job_id = sys.argv[1]	
	country = sys.argv[2]
	if len(sys.argv) > 2:
		color = sys.argv[3]
		if color in colors:
			color = colors[color]
	else:
		color = blue
	
	X, T = load_data(country, up_to_date=last_date)
	idx = date_threshold < T 
	ndays = len(X)
	
	chain_fname = os.path.join(output_folder, job_id, 'inference', '{}.npz'.format(country))
	chain, _, Td1, Td2, model_type, _, start_date, N = load_chain(fname=chain_fname)
	X_mean = scipy.signal.savgol_filter(X, 3, 1)
	
	model_class = get_model_class(model_type)
	model = model_class(country, X, pd.to_datetime(start_date), N, get_last_NPI_date(country), get_first_NPI_date(country), params_bounds, Td1, Td2)
	X_pred = posterior_prediction(chain, model, nreps)


	pvalue = (X_pred[:,idx].max(axis=1) > X[idx].max()).mean() # P(max(X_pred) > max(X))
	pvalue_file = os.path.join(output_folder, job_id, 'figures', 'ppc_pvalue.txt'.format(country))
	with open(pvalue_file, 'at') as f:
		print("{}\t{:.4g}".format(country, pvalue), file=f)

	fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
	ymax = min(X.max()*2, max(X.max(), X_pred.max()))

	t = np.arange(0, ndays)
	ax.plot(t[~idx], X[~idx], 'o', color='k', alpha=0.5)
	ax.plot(t[~idx], X_mean[~idx], '-', color='k')
	ax.plot(t[idx], X[idx], '*', color='k', alpha=0.5)
	ax.plot(t[idx], X_mean[idx], '--', color='k')
	    
	ax.plot(X_pred.T, color=color, alpha=0.01)
	
	ax.axvline((date_threshold-pd.to_datetime(start_date)).days, color='k', ls='--', lw=2)

	labels = [τ_to_string(int(d), start_date) for d in t[::5]]	
	ax.set_xticks(t[::5])
	ax.set_xticklabels(labels, rotation=45)
	ax.set(ylabel='Daily cases', ylim=(-10, ymax))

	# fig.suptitle(country.replace('_', ' '))
	fig.tight_layout()
	sns.despine()
	# plt.show()
	fig_filename = os.path.join(output_folder, job_id, 'figures', '{}_ppc_long.pdf'.format(country))
	print("Saving to {}".format(fig_filename))
	fig.savefig(fig_filename)
