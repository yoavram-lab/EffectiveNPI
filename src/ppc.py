#!/usr/bin/env python
# coding: utf-8
import sys
import os
from datetime import datetime, timedelta
import urllib

import matplotlib as mpl
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


def load_data(country):
	url = 'https://github.com/ImperialCollegeLondon/covid19model/raw/v1.0/data/COVID-19-up-to-date.csv'
	fname = '../data/COVID-19-up-to-date.csv'
	if not os.path.exists(fname):
	    urllib.request.urlretrieve(url, fname)
	data = pd.read_csv(fname, encoding='iso-8859-1')
	data['date'] = pd.to_datetime(data['dateRep'], format='%d/%m/%Y')
	data = data[data['countriesAndTerritories'] == country]
	N = data.iloc[0]['popData2018']
	cases_and_dates = data.iloc[::-1][['cases','date']]
	start_date = find_start_day(cases_and_dates)
	X = np.array(cases_and_dates[cases_and_dates['date'] >= start_date]['cases'])	
	return X, start_date, N

def load_chain(job_id):
	with spinner():
		fname = os.path.join(output_folder, job_id, 'inference', '{}.npz'.format(country))
		inference_data = np.load(fname)
		chain = inference_data['chain']
		var_names = inference_data['var_names']
		# print("Loaded {} with parameters:".format(fname))
		# print(var_names)
		nchains, nsteps, ndim = chain.shape
		chain = chain[:, nsteps//2:, :]
		chain = chain.reshape((-1, ndim))
		return chain, var_names


def posterior_prediction(chain, model, nreps):
	θ = chain[np.random.choice(chain.shape[0], nreps)]
	return np.array([
		model.generate_daily_cases(θi) for θi in θ
	])



if __name__ == '__main__':
	nreps = 1000
	output_folder = r'/Users/yoavram/Library/Mobile Documents/com~apple~CloudDocs/EffectiveNPI-Data/output/'
	job_id = sys.argv[1]	
	country = sys.argv[2]
	if len(sys.argv) > 2:
		color = sys.argv[3]
		if color in colors:
			color = colors[color]
	else:
		color = blue
	X, start_date, N = load_data(country)
	ndays = len(X)
	X_mean = scipy.signal.savgol_filter(X, 3, 1)

	chain, var_names = load_chain(job_id)
	
	if 'τ' in var_names:
		model_class = NormalPriorModel
	else:
		model_class = FixedTauModel
	model = model_class(country, X, start_date, N, get_last_NPI_date(country), get_first_NPI_date(country), params_bounds, 9, 6)
	X_pred = posterior_prediction(chain, model, nreps)

	ϵ = 1
	loss = ((X_pred - X_mean + ϵ)**2 / (X_pred + ϵ)**2).mean()

	print("Loss: {:.2g}".format(loss))

	fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)

	t = np.arange(0, ndays)
	ax.plot(t, X, '*', color='k', alpha=0.5)
	ax.plot(t, X_mean, '-', color='k')
	    
	ax.plot(X_pred.T, color=color, alpha=0.01)
	
	labels = [τ_to_string(int(d), start_date) for d in t[::5]]	
	ax.set_xticks(t[::5])
	ax.set_xticklabels(labels, rotation=45)
	ax.set(ylabel='Daily cases')	

	# fig.suptitle(country.replace('_', ' '))
	fig.tight_layout()
	sns.despine()
	plt.show()
	fig_filename = os.path.join(output_folder, job_id, 'figures', '{}_ppc.pdf'.format(country))
	print("Saving to {}".format(fig_filename))
	fig.savefig(fig_filename)
