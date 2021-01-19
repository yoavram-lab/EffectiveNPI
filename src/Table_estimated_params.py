#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys

sns.set_context('paper', font_scale=1.3)
red, blue, green = sns.color_palette('Set1', 3)

def int_to_dt(t):
    return pd.to_datetime(start_date) + timedelta(days=t)

if __name__ == '__main__':
	job_id = sys.argv[1]
	output_folder = r'/Users/adsurbum/projects/univer/EffectiveNPI/output-tmp/{}/'.format(job_id)

	NPI_dates = pd.read_csv('../data/NPI_dates.csv')

	data = pd.read_csv(os.path.join(output_folder, 'tables', 'all-countries-{}.csv'.format(job_id)))
	features = ['country', 'official_τ','τ median d', 'τ median', 'τ hpd 75% from n', 'τ hpd 75% to n','τ hpd 95% from n', 'τ hpd 95% to n']
	features += [x for x in data.columns if 'median' in x and 'τ' not in x and 'loglik' not in x and 'DIC' not in x]

	table = data[features].copy()
	table['country'] = [x.replace('_', ' ') for x in table['country']]
	table.columns = [x.replace(' median', '') for x in table.columns]
	table.insert(3,'hdi75f', table['τ hpd 75% from n']-table['τ'])
	table.insert(4,'hdi75t', table['τ hpd 75% to n']-table['τ'])
	table.insert(5,'hdi95f', table['τ hpd 95% from n']-table['τ'])
	table.insert(6,'hdi95t', table['τ hpd 95% to n']-table['τ'])
	table =table.drop(columns=['τ hpd 75% from n','τ hpd 75% to n','τ hpd 95% from n','τ hpd 95% to n','τ'])
	table = table.rename(columns=
		{'country': 'Region', 'official_τ': r'$\tau^*$', 'τ d' : r'$\hat{\tau}$', 
		'hdi75f': '$HDI_{75\%}$', 'hdi75t':'', 'hdi95f': '$HDI_{95\%}$','hdi95t' : '', 
		'Δt0': r'$\Delta t$', 'E0': r'$E(0)$', 'Iu0': r'$I_u(0)$', 'μ': r'$\mu$', 'β': r'$\beta$', 'λ': r'$\lambda$', 
		'α1': r'$\alpha_1$', 'α2': r'$\alpha_2$', 'Z': '$Z$', 'D': '$D$'
		})
	table = table.sort_values('Region')
	table.loc[table['Region'] == 'Wuhan', 'Region'] = 'Wuhan, China'
	table.columns = ["\ccell{"+x+"}" for x in table.columns]

	csv_filename = '../figures/Table-estimated-params.csv'
	tex_filename = '../figures/Table-estimated-params.tex'
	table.to_csv(csv_filename, index=False)
	with open(tex_filename, 'wt') as f:
		print(table.to_latex(escape=False, index=False), file=f)
	print("Saved output to {} and {}".format(csv_filename, tex_filename))