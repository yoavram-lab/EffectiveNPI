#!/usr/bin/env python
# coding: utf-8
import os
from datetime import datetime, timedelta
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from arviz import hpd

from rakott.mpl import savefig_bbox

sns.set_context('paper', font_scale=1.3)
red, blue, green, purple = sns.color_palette('Set1', 4)
colors = {'red':red, 'blue':blue, 'green':green, 'purple':purple}

def int_to_dt(t):
    return pd.to_datetime(start_date) + timedelta(days=t)

def calc_hpd(trace, mass_frac) :
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])

if __name__ == '__main__':
	job_id = sys.argv[1]
	output_folder = r'../output-tmp/{}/'.format(job_id)
	country = sys.argv[2]
	quiet = len(sys.argv) > 3 and sys.argv[3] == '-q'	
	nburn = 2_000_000

	NPI_dates = pd.read_csv('../data/NPI_dates.csv')

	if country == 'all':
		countries = [country.replace(' ', '_') for country in NPI_dates['Country']]
	else:
		countries = [country]

	if country=='Spain':
		delete_chain_less_than = 15 #TODO add as input parameter
	else:
		delete_chain_less_than = None

	for country in countries:
		npz_path = os.path.join(output_folder, 'inference', '{}.npz').format(country)
		print("Loading inference data from", npz_path)
		data = np.load(npz_path)

		var_names = list(data['var_names'])
		start_date = data['start_date']
		start_date = pd.to_datetime(start_date)
		sample = data['chain']
		log_posterior = data['lnprobability']

		if delete_chain_less_than:
			if len((sample[:,1_000_000, var_names.index('τ')]<delete_chain_less_than).nonzero())>1:
				raise AssertionError('too many bad chains')
			bad_chain_ind = (sample[:,1_000_000, var_names.index('τ')]<delete_chain_less_than).nonzero()[0][0]
			sample = np.delete(sample, bad_chain_ind, axis=0)
			log_posterior = np.delete(log_posterior, bad_chain_ind, axis=0)

		first_date = pd.to_datetime(NPI_dates.loc[NPI_dates['Country'] == country.replace('_', ' '), 'First'].values[0])
		last_date = pd.to_datetime(NPI_dates.loc[NPI_dates['Country'] == country.replace('_', ' '), 'Last'].values[0])
		first_date_days = (first_date - start_date).days
		last_date_days = (last_date - start_date).days

		τ_sample = sample[:, :, -1]

		thin = 100
		plt.plot(τ_sample[:, ::thin].T)
		plt.ylabel('τ')
		plt.xlabel('Iteration')
		plt.title('Trace plot: {}'.format(country));
		plt.axvline(nburn/thin, color='k');
		if not quiet: plt.show()

		plt.plot(τ_sample[:,nburn::thin].T)
		plt.ylabel('τ')
		plt.xlabel('Iteration')
		plt.title('Trace plot (burnin): {}'.format(country));
		if not quiet: plt.show()

		fig, ax = plt.subplots()
		for i in range(τ_sample.shape[0]):
		    ax.hist(τ_sample[i, nburn:], density=False, alpha=0.4)
		ax.set(xlabel='τ', ylabel='Posterior probability')
		ax.set_title('Posterior: {}'.format(country))
		if not quiet: plt.show()

		idx_max = log_posterior[:, nburn:].ravel().argmax()
		τ_sample = τ_sample[:, nburn:].ravel()
		τ_map = τ_sample[idx_max]
		print("τ MAP = {:.2f}".format(τ_map), int_to_dt(τ_map))
		τ_mean = np.mean(τ_sample)
		print("τ mean = {:.2f}".format(τ_mean), int_to_dt(τ_mean))
		τ_med = np.median(τ_sample)
		print("τ median = {:.2f}".format(τ_med), int_to_dt(τ_med))
		quantile = 0.95
		τ_ci = np.quantile(abs(τ_sample - τ_med), quantile)
		print("+-{:.2f} days for {:.2f} quantile".format(τ_ci, quantile))
		# hpi_ = hpd(τ_sample, quantile)#, multimodal=True)
		# print("HPI {:.2f}-{:.2f} for {:.2f} confidence".format(*hpi_, quantile))
		τ_hpd_from, τ_hpd_to = calc_hpd(τ_sample, quantile)
		# print(τ_hpd_from,τ_hpd_to)
		# print(hpd(τ_sample, quantile)) #arviz hpd is the same

		fig, ax = plt.subplots(figsize=(6, 3))
		xmin = τ_sample.min()-2.5
		xmax = max(last_date_days, np.ceil(τ_sample.max())) + 2.5
		density, bins, _ = ax.hist(τ_sample, bins=np.arange(0, xmax, 1), density=True, align='mid')
		ymax = density.max() * 1.2
		ax.axvline(τ_med, color='k')
		# ax.fill_between([τ_med - τ_ci, τ_med + τ_ci], 0, ymax, alpha=0.4, color='k')
		ax.fill_between([τ_hpd_from, τ_hpd_to], 0, ymax, alpha=0.4, color='k')
		# for hpi_ in hpis:
		# ax.fill_between([hpi_[0], hpi_[1]], 0, ymax, alpha=0.4, color='k')
		# ax.axvline(first_date_days, color=red)
		ax.axvline(last_date_days, color=red)		
		# ax.fill_between([first_date_days, last_date_days], 0, ymax, alpha=0.4, color=red)
		ax.set(xlabel=r'Effective start of NPI, $\tau$', 
		       ylabel=r'Posterior density, $P(\tau \mid \vec{X})$', 
		       xlim=(τ_sample.min()-1, τ_sample.max()+1),
		       ylim=(0, ymax)
		)
		days = np.arange(xmin, xmax, 3)
		deltas = [timedelta(int(x)) for x in days]
		ax.set_xticks(days);
		txt = ax.set_xticklabels([(start_date + d).strftime('%b %d') for d in deltas], rotation=45)

		ax.annotate(r'$\hat{\tau}$', (τ_med+0.1, ymax*0.875))
		# ax.annotate(r'$\tau^0$', (first_date_days+0.1, ymax*0.875))
		ax.annotate(r'$\tau^*$', (last_date_days+0.1, ymax*0.875))

		ax.set_title(country.replace('_', ' '))
		sns.despine()
		if not quiet: plt.show()
		fig_filename = os.path.join(output_folder, 'figures', '{}_τ_posterior_hpd.pdf'.format(country))
		print("Saving to", fig_filename)
		fig.savefig(fig_filename, dpi=100, **savefig_bbox(*txt))
