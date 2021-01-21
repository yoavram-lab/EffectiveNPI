#!/usr/bin/env python
# coding: utf-8
import sys
import os
from datetime import datetime, timedelta
import urllib

import matplotlib as mpl
# mpl.use("Agg")
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
from sklearn.metrics import mean_squared_error
from Fig_ppc import load_chain

def posterior_Re(chain, nreps):
	# params: ['Z', 'D', 'μ', 'β', 'α1', 'λ', 'α2', ...]
	θ = chain[np.random.choice(chain.shape[0], nreps)]
	Re_pre = Re(θ[:, 4], θ[:, 3], θ[:, 1], θ[:, 2])
	Re_post = Re(θ[:, 6], θ[:, 3]*θ[:, 5], θ[:, 1], θ[:, 2])
	return Re_pre, Re_post

def Re(α, β, D, μ):
    # Li et al 2020, SI pg 4
    return α*β*D + (1-α)*μ*β*D 

if __name__ == '__main__':
	nreps = 1000
	
	output_folder = r'../output'
	job_id = sys.argv[1]	
	country = sys.argv[2]
	if len(sys.argv) > 3:
		nreps = int(sys.argv[3])
	
	Re_file = os.path.join(output_folder, job_id, 'figures', 'Re_{}.csv'.format(country))
	if os.path.exists(Re_file):
		print(Re_file, "already exists")
	else:
		chain_fname = os.path.join(output_folder, job_id, 'inference', '{}.npz'.format(country))
		delete_chain_less_than = None
		if job_id=='7M' and country=='Spain':  #TODO make input parameter
			delete_chain_less_than = 15 
		chain, _, _, _, model_type, _, _, _ = load_chain(fname=chain_fname)
		Re_pre, Re_post = posterior_Re(chain, nreps)
		rel_reduc_Re = 1 - Re_post / Re_pre
		df = pd.DataFrame(dict(Re_pre=Re_pre, Re_post=Re_post, rel_reduc_Re=rel_reduc_Re))	
		df.to_csv(Re_file, index=False)
		print(Re_file)
