import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
sns.set_context('paper', font_scale=1.3)
import sys

def plot_τ_trace(path, country_name):
    file_name_npz = '{}/inference/{}.npz'.format(path, country_name)
    data = np.load(file_name_npz)
    chain = data['chain']
    var_names = list(data['var_names'])
    
    fig, ax = plt.subplots()
    plt.plot(chain[:,::100,var_names.index('τ')].T, color='k', alpha=0.15);
    labels = ['−10000', '0', '1', '2', '3', '4', '5', '6', '7', '8']
    ax.set_xticklabels(labels);
    ax.text(x = 0.93, y = -0.15, s = '1e6', transform=ax.transAxes);
    plt.xlim(-2000, 70000);
    plt.xlabel('number of iterations')
    plt.ylabel('τ')
    plt.title(country_name);
    plt.tight_layout()
    fig.savefig('{}/figures/{}_trace1.png'.format(path, country_name),dpi=100)
    
    fig, ax = plt.subplots()
    plt.plot(chain[:,::100,var_names.index('τ')].T, color='k', alpha=0.05);
    labels = ['−10000', '0', '1', '2', '3', '4', '5', '6', '7', '8']
    ax.set_xticklabels(labels);
    ax.text(x = 0.93, y = -0.15, s = '1e6', transform=ax.transAxes);
    plt.xlim(-2000, 70000);
    plt.xlabel('number of iterations')
    plt.ylabel('τ')
    plt.title(country_name);
    plt.tight_layout()
    fig.savefig('{}/figures/{}_trace2.png'.format(path, country_name),dpi=100)
    
    fig, ax = plt.subplots()
    plt.plot(chain[:,::100,var_names.index('τ')].T);
    # fig.canvas.draw()
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ['−10000', '0', '1', '2', '3', '4', '5', '6', '7', '8']
    ax.set_xticklabels(labels);
    ax.text(x = 0.93, y = -0.15, s = '1e6', transform=ax.transAxes);
    plt.xlim(-2000, 70000);
    plt.xlabel('number of iterations')
    plt.ylabel('τ')
    plt.title(country_name);
    plt.tight_layout()
    fig.savefig('{}/figures/{}_trace3.png'.format(path, country_name),dpi=100)
    
path = './'
job_id = sys.argv[1]
country = sys.argv[2]
plot_τ_trace('{}/{}'.format(path,job_id), country)