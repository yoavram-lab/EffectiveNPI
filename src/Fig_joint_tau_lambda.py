import os,sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

sns.set_context('paper', font_scale=1.6)

def hex_wrapper(*args, **kwargs):
    kwargs['color'] = None
    return plt.hexbin(*args,**kwargs)
    
def τ_to_string(τ,start_date):
    return (pd.to_datetime(start_date) + timedelta(days=τ)).strftime('%b %d')

def plot_joint_τ_λ(path, country_name):
    file_name_npz = '{}/inference/{}.npz'.format(path, country_name)
    data = np.load(file_name_npz)
    chain = data['chain']
    var_names = list(data['var_names'])
    nsteps, ndim, N, Td1, Td2, model_type = data['params']
    start_date = data['start_date']
    τ_max = len(data['incidences'])-2
    τ_min = 10

    #TODO add parameter
    if country_name=='Spain':
        delete_chain_less_than=15
        if len((chain[:,1_000_000, var_names.index('τ')]<delete_chain_less_than).nonzero())>1:
            raise AssertionError('too many bad chains')
        bad_chain_ind = (chain[:,1_000_000, var_names.index('τ')]<delete_chain_less_than).nonzero()[0][0]
        chain = np.delete(chain, bad_chain_ind, axis=0)

    sample = chain[:, 2_000_000:, :].reshape(-1, ndim)
    
    np.random.seed(10)
    randindxs = np.random.choice(len(sample),20000,replace=False)
    
    def plotone(s=0):
        post_sample = sample[randindxs]
        if s<2:
            post_sample[0,var_names.index('λ')] = 0
            post_sample[1,var_names.index('λ')] = 1
        if s<1:
            post_sample[0,var_names.index('τ')] = τ_min #should be problematic for yticks if we don't do it, so need to change yticks
            post_sample[1,var_names.index('τ')] = τ_max
        df = pd.DataFrame(post_sample)
        df.columns=var_names    

        g = sns.PairGrid(df, y_vars=["τ"], x_vars=['λ'], height=6)
        cmap = sns.cubehelix_palette(8,start=10,light=1, as_cmap=True)
        g = g.map(hex_wrapper, gridsize=47,cmap=cmap) #'Greys'
        g.fig.suptitle(country_name)
        days = np.linspace(τ_min,τ_max,5)
        labels = [τ_to_string(d,start_date) for d in days]
        plt.yticks(days,labels);
        plt.ylabel(r"τ");
        plt.tight_layout()
        return g.fig
    
    if not os.path.exists('{}/figures/joint'.format(path)):
        os.makedirs('{}/figures/joint'.format(path))

    plotone(0).savefig('{}/figures/joint/{}_joint.pdf'.format(path, country_name))
    # plotone(1).savefig('{}/figures/joint/{}_joint2.pdf'.format(path, country_name))
    # plotone(2).savefig('{}/figures/joint/{}_joint3.pdf'.format(path, country_name))

path = '.'
job_id = sys.argv[1]
country = sys.argv[2]
fig = plot_joint_τ_λ('{}/{}'.format(path,job_id), country)
