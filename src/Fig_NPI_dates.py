#!/usr/bin/env python
# coding: utf-8
import os
from datetime import datetime, timedelta

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rakott.mpl import savefig_bbox

def int_to_dt(t):
    return pd.to_datetime(start_date) + timedelta(days=t)
def dt_to_days(t):
    return (t - datetime(2020, 1, 1)).days
def str_to_dt(t):
    return datetime.strptime(t, '%b %d %Y')

sns.set_context('paper', font_scale=1.3)
red, blue, green = sns.color_palette('Set1', 3)
colors = sns.color_palette('Paired')

date_range = [datetime(2020, 3, 1) + timedelta(weeks=i) for i in range(5)]
date_formatter = mpl.dates.DateFormatter('%b %d')

def get_official_dates():
	official = pd.read_csv('../data/NPI_dates.csv').sort_values('Country', ascending=False)
	official['First'] = [str_to_dt(t) for t in official['First']]
	official['Last'] = [str_to_dt(t) for t in official['Last']]
	official['First (days)'] = [dt_to_days(t) for t in official['First']]
	official['Last (days)'] = [dt_to_days(t) for t in official['Last']]
	return official

if __name__ == '__main__':

	official = get_official_dates()

	fig, ax = plt.subplots(figsize=(4, 6))
	data_ = official[official['First'] >= datetime(2020, 3, 1)]
	ax.plot('First', 'Country', 'ok', data=data_)
	ax.plot('Last', 'Country', 'ok', data=data_)
	ax.hlines('Country', 'First', 'Last', data=data_)
	ax.set_xticks(date_range)
	ax.xaxis.set_major_formatter(date_formatter)
	ax.grid(True)
	txt = ax.get_yticklabels()
	sns.despine()
	fig.savefig('../figures/Fig-NPI_dates.pdf', dpi=100, **savefig_bbox(*txt))
	print("Saved NPI dates to ../figures/Fig-NPI_dates.pdf")
