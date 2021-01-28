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
    file_name = '7MFixed/inference/{}.autocorr'.format(country_name)
    auto = np.loadtxt(file_name)
    iters = np.array([0]+[i*200000 for i in range(1,len(auto)+1)])/1_000_000
    taus = np.array([0]+list(auto))/1000
    print(int(50*(200_000*len(auto)-2_000_000)/1000/max(taus)),'\t',int(max(taus)),'\t', 200_000*len(auto), country_name)
    plt.plot(iters, taus, label=country_name)
    plt.xlabel("Samples (1M)")
    plt.ylabel("Average IAT (1K)");
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(0,7)
    return lgd

for c in ['Austria','Belgium','Denmark','France','Germany','Italy','Norway',
          'Spain','Sweden','Switzerland','United_Kingdom','Wuhan']:
    lgd = plot_auto(c)
plt.ylim(0)

# plt.gca().xaxis.set_major_formatter(FixedScalarFormatter(6)) #doesn't work :\\\\
# plt.gca().yaxis.set_major_formatter(FixedScalarFormatter(3)) 
# plt.tight_layout()
plt.gcf().savefig('7MFixed/figures/Fig-autocorr.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')