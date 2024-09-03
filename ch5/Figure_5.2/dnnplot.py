import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import gaussian_kde
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
pyrate = np.load('./data/pyrate.npy')
nnrate = np.load('./data/nnrate.npy')
mprate = np.load('./data/mprate.npy')
rdrate = np.load('./data/rdrate.npy')

plt.figure()
plt.style.use('seaborn-deep')
data = np.vstack([pyrate, nnrate, mprate, rdrate]).T
#pyrate_data = np.sort(pyrate)
sorted_data = np.sort(data, axis=0)
#bins = np.linspace(0, max(pyrate), 50)
cumulative_percentiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
linestyle = ['-.',':','--','-']
markers = ['o', 's', 'D', '^']
label = ['WMMSE', 'DNN', 'MAX Power', 'Random Power']
for i in range(len(linestyle)):
    plt.plot(sorted_data[:, i], cumulative_percentiles, linestyle=linestyle[i], label=label[i], marker=markers[i], markevery=0.1, linewidth=2.0)

plt.legend(loc='upper right')
plt.xlim([0, 8])
plt.xlabel('sum-rate (bit/sec)', fontsize = 14)
plt.ylabel('cumulative percentiles', fontsize = 14)
plt.savefig('fig5_ch5_CDF', format='eps', dpi=2000)
plt.show()

plt.figure()
plt.style.use('seaborn-deep')
data = np.vstack([pyrate, nnrate]).T
bins = np.linspace(0, max(pyrate), 50)
linestyle = ['-','--']
label = ['WMMSE', 'DNN']
markers = ['o', 's']
# for i in range(len(linestyle)):
#     plt.hist(data[:, i], bins, label=label[i], alpha=0.7, histtype='bar')
plt.hist(data, bins, alpha=0.7, label=['WMMSE', 'DNN'],)
plt.legend(loc='upper right')
plt.xlim([0, 8])
plt.xlabel('sum-rate (bit/sec x Hz)', fontsize=14)
plt.ylabel('number of samples', fontsize=14)
plt.savefig('Histogram_%d.eps', format='eps', dpi=2000)
plt.show()

