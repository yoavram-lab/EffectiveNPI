#!/usr/bin/env python
# coding: utf-8
import os
from datetime import datetime, timedelta
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rakott.mpl import savefig_bbox

from Fig_NPI_dates import get_official_dates

def int_to_dt(t):
    return pd.to_datetime(start_date) + timedelta(days=t)
def dt_to_days(t):
    return (t - datetime(2020, 1, 1)).days
def str_to_dt(t):
    return datetime.strptime(t, '%b %d %Y')
def date_to_int(x):
    dt = datetime.strptime(x + ' 2020', '%b %d %Y')
    td = dt - datetime(2020, 1, 1)
    return td.days
def date_to_date(x):
    dt = datetime.strptime(x + ' 2020', '%b %d %Y')
    return dt

sns.set_context('paper', font_scale=1.3)
red, blue, green = sns.color_palette('Set1', 3)
colors = sns.color_palette('Paired')

date_range = [datetime(2020, 3, 1) + timedelta(weeks=i) for i in range(5)]
date_formatter = mpl.dates.DateFormatter('%b %d')


if __name__ == '__main__':
	job_id = sys.argv[1]
	verbose = len(sys.argv) > 2 and sys.argv[2] == '-v'

	output_folder = r'/Users/yoavram/Library/Mobile Documents/com~apple~CloudDocs/EffectiveNPI-Data/output/{}'.format(job_id)
	
	official = get_official_dates()
	table_path = os.path.join(output_folder, 'tables', 'all-countries-{}.csv'.format(job_id))
	df = pd.read_csv(table_path)
	df = df.rename(columns={'official_τ': 'τ official'})
	df['country'] = df['country'].str.replace("_", " ")

	countries = df['country'].unique()
	country_color = {country: colors[i] for i, country in enumerate(countries)}
	df['color'] = [country_color[country] for country in df['country']]

	df['τ mean days'] = [date_to_int(x) for x in df['τ mean']]
	df['τ median days'] = [date_to_int(x) for x in df['τ median']]
	df['τ MAP days'] = [date_to_int(x) for x in df['τ MAP']]
	df['τ official days'] = [date_to_int(x) for x in df['τ official']]
	df['τ official - mean days'] = df['τ official days'] - df['τ mean days']
	df['τ official - median days'] = df['τ official days'] - df['τ median days']
	df['τ median - official days'] = df['τ median days'] - df['τ official days']
	df['τ official - MAP days'] = df['τ official days'] - df['τ MAP days']
	df['τ mean'] = [date_to_date(x) for x in df['τ mean']]
	df['τ median'] = [date_to_date(x) for x in df['τ median']]
	df['τ MAP'] = [date_to_date(x) for x in df['τ MAP']]
	df['τ official'] = [date_to_date(x) for x in df['τ official']]

	if verbose:
		sns.regplot('τ median from 1 Jan', 'τ mean from 1 Jan', data=df, ci=False)
		plt.plot(np.arange(30, 90), np.arange(30, 90), ls='--', color='k')
		print("Correlation: {} and {}".format('τ median from 1 Jan', 'τ mean from 1 Jan'))
		print(df[['τ median from 1 Jan', 'τ mean from 1 Jan']].corr())
		plt.show()

		print("Correlation: {} and {}".format('τ median from 1 Jan', 'τ MAP from 1 Jan'))
		print(df[['τ median from 1 Jan', 'τ MAP from 1 Jan']].corr())
		sns.regplot('τ median from 1 Jan', 'τ MAP from 1 Jan', data=df, ci=False)
		plt.plot(np.arange(30, 90), np.arange(30, 90), ls='--', color='k')
		plt.show()

	col = 'τ median - official days'
	ci75_col = 'τ CI median (75%)'
	ci95_col = 'τ CI median (95%)'

	fig, ax = plt.subplots(figsize=(6, 6))
	df_ = df.sort_values(col, ascending=False)

	val = df_[col]
	ci75 = df_[ci75_col]
	ci95 = df_[ci95_col]
	country = df_['country']
	ax.hlines(country, val-ci95, val+ci95)
	ax.hlines(country, val-ci75, val+ci75, lw=3)

	# idx = df_[col] >= 1
	idx = df_[col] + ci75 < 0
	val = df_.loc[idx, col]
	country = df_.loc[idx, 'country']
	ax.plot(val, country, 'o', markersize=10, color=blue)


	# # idx = df_[col] <= -1
	idx = 0 < df_[col] - ci75
	val = df_.loc[idx, col]
	country = df_.loc[idx, 'country']
	ax.plot(val, country, 'o', markersize=10, color=red)

	# # idx = (-1 < df_[col]) & (df_[col] < 1)
	idx = (0 <= df_[col] + ci75) & (df_[col] - ci75 <= 0)
	val = df_.loc[idx, col]
	country = df_.loc[idx, 'country']
	ax.plot(val, country, 'o', markersize=10, color='k')

	plt.axvline(0, ls='--', color='k')
	ax.set(
	    xlabel=r'Days between effective and official date, $\hat{\tau} - \tau^*$',
	)
	ax.annotate('Late', (7, 11.5), fontsize=16)
	ax.annotate('Early', (-12, 11.5), fontsize=16)
	sns.despine()
	plt.tight_layout()
	if verbose: plt.show()
	fig.savefig('../figures/Fig-tau-summary.pdf', dpi=100)
	print("Saved to ../figures/Fig-tau-summary.pdf")

