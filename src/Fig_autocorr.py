import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
sns.set_context('paper', font_scale=1.3)

class FixedScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, power, useOffset=None, useMathText=None, useLocale=None):
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset, useMathText, useLocale)
        self.power = power
        self.set_scientific(True)
#         self.set_powerlimits((2,6))

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.power
        
def plot_auto(country_name): 
    file_name = '7M/inference/{}.autocorr'.format(country_name)
    auto = np.loadtxt(file_name)
    iters = np.array([0]+[i*200000 for i in range(1,len(auto)+1)])
    taus = np.array([0]+list(auto))
#     plt.plot(iters, iters/100.0,'--k')
    plt.plot(iters, taus, label=country_name)
    plt.xlabel("number of iterations")
    plt.ylabel(r"auto-correlation mean");
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(0,iters[-1])

for c in ['Austria','Belgium','Denmark','France','Germany','Italy','Norway',
          'Spain','Sweden','Switzerland','United_Kingdom','Wuhan']:
    plot_auto(c)
plt.ylim(0)

plt.gca().xaxis.set_major_formatter(FixedScalarFormatter(6)) #doesn't work :\\\\
plt.gca().yaxis.set_major_formatter(FixedScalarFormatter(3)) 
plt.gcf().savefig('7M/figures/auto-correlation.pdf')