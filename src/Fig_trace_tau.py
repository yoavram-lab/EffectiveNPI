import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import sys
from datetime import timedelta, datetime
import pandas as pd

sns.set_context('paper', font_scale=1.6)


def τ_to_string(τ,start_date):
    return (pd.to_datetime(start_date) + timedelta(days=τ)).strftime('%b %d')

def plot_τ_trace(path, country_name, thin=10_000, alpha=0.1):
    file_name_npz = '{}/inference/{}.npz'.format(path, country_name)
    data = np.load(file_name_npz)
    chain = data['chain']
    var_names = list(data['var_names'])
    start_date = data['start_date']
    maxx = len(data['incidences'])-2
    minn = 10
    
    fig, ax = plt.subplots()
    plt.plot(chain[:,::thin,var_names.index('τ')].T, alpha=alpha);
    labels = ['−10000', '0', '1', '2', '3', '4', '5', '6', '7', '8']
    ax.set_xticklabels(labels);
    plt.xlim(-2_00_000/thin, 7_000_000/thin);
    plt.ylim(minn,maxx)
    plt.xlabel('Samples (1M)')
    plt.ylabel('τ')
    plt.title(country_name);
    days = np.linspace(minn,maxx,5)
    labels = [τ_to_string(d,start_date) for d in days]
    plt.yticks(days,labels);
    plt.tight_layout()
    fig.savefig('{}/figures/{}_trace.pdf'.format(path, country_name),dpi=100)
    
path = './'
job_id = sys.argv[1]
country = sys.argv[2]
plot_τ_trace('{}/{}'.format(path,job_id), country)