import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from datetime import datetime, timedelta
from rakott.mpl import savefig_bbox
import sys

sns.set_context('paper', font_scale=1.3)
red, blue, green = sns.color_palette('Set1', 3)

def int_to_dt(t):
    return pd.to_datetime(start_date) + timedelta(days=t)

if __name__ == '__main__':
	job_id = sys.argv[1]
	output_folder = r'/Users/yoavram/Library/Mobile Documents/com~apple~CloudDocs/EffectiveNPI-Data/output/{}/'.format(job_id)
	country = sys.argv[2]

	NPI_dates = pd.read_csv('../data/NPI_dates.csv')

	if country == 'all':
		countries = [country.replace(' ', '_') for country in NPI_dates['Country']]
	else:
		countries = [country]

	for country in countries:
		npz_path = os.path.join(output_folder, 'inference', '{}.npz').format(country)
		print("Loading inference data from", npz_path)
		data = np.load(npz_path)

		var_names = data['var_names']
		start_date = data['start_date']
		start_date = pd.to_datetime(start_date)
		sample = data['chain']
		log_posterior = data['lnprobability']

		official_date = pd.to_datetime(NPI_dates.loc[NPI_dates['Country'] == country.replace('_', ' '), 'Last'].values[0])

		official_date_days = (official_date - start_date).days

		τ_sample = sample[:, :, -1]
		nburn = τ_sample.shape[1]//2

		thin = 100
		plt.plot(τ_sample[:, ::thin].T)
		plt.ylabel('τ')
		plt.xlabel('Iteration')
		plt.title('Trace plot: {}'.format(country));
		plt.axvline(nburn/thin, color='k');
		plt.show()

		plt.plot(τ_sample[:,nburn::thin].T)
		plt.ylabel('τ')
		plt.xlabel('Iteration')
		plt.title('Trace plot (burnin): {}'.format(country));
		plt.show()

		fig, ax = plt.subplots()
		for i in range(τ_sample.shape[0]):
		    ax.hist(τ_sample[i, nburn:], density=False, alpha=0.4)
		ax.set(xlabel='τ', ylabel='Posterior probability')
		ax.set_title('Posterior: {}'.format(country))
		plt.show()

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

		fig, ax = plt.subplots(figsize=(6, 3))
		xmax = max(official_date_days, np.ceil(τ_sample.max())) + 2.5
		density, bins, _ = ax.hist(τ_sample, bins=np.arange(0, xmax, 1), density=True, align='mid')
		ymax = density.max() * 1.2
		ax.axvline(τ_med, color='k')
		ax.fill_between([τ_med - τ_ci, τ_med + τ_ci], 0, ymax, alpha=0.4, color='k')
		ax.axvline(official_date_days, color=red)
		ax.set(xlabel=r'Effective start of NPI, $\tau$', 
		       ylabel=r'Posterior density, $(P(\tau \mid \vec{X})$', 
		       xlim=(τ_sample.min()-1, τ_sample.max()+1),
		       ylim=(0, ymax)
		)
		days = np.arange(0, xmax, 3)
		deltas = [timedelta(int(x)) for x in days]
		ax.set_xticks(days);
		txt = ax.set_xticklabels([(start_date + d).strftime('%b %d') for d in deltas], rotation=45)

		ax.annotate(r'$\hat{\tau}$', (τ_med+0.1, ymax*0.875))
		ax.annotate(r'$\tau^*$', (official_date_days+0.1, ymax*0.875))

		ax.set_title(country.replace('_', ' '))
		sns.despine()
		plt.show()
		fig_filename = os.path.join(output_folder, 'figures', '{}_τ_posterior.pdf'.format(country))
		print("Saving to", fig_filename)
		fig.savefig(fig_filename, dpi=100, **savefig_bbox(*txt))
