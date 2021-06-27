#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

# first array contains tesing score, 
# second contains one percent true, 
# third contains one percent false
# data = [[90.06, 90.05, 90.04, 89.96, 89.95, 89.91, 89.87, 89.73, 89.00, 85.10],
#         [88.55, 88.27, 88.02, 88.83, 88.28, 88.77, 88.02, 88.34, 88.02, 87.11],
#         [91.01, 91.21, 91.91, 90.91, 91.14, 90.61, 91.91, 91.02, 90.78, 82.84]]

# error = [[0.0013, 0.0016, 0.0048, 0.0016, 0.0013, 0.0017, 0.0046, 0.0011, 0.0051, 0.0022],
#          [0.0015, 0.0008, 0.0015, 0.0011, 0.0038, 0.0012, 0.0009, 0.0009, 0.0048, 0.0016],
#          [0.0013, 0.0009, 0.0044, 0.0011, 0.0009, 0.0005, 0.0046, 0.0008, 0.0046, 0.0020]]

#first array contains full phonemes, second contains half
# data = [[82.82, 87.91, 87.83, 87.86], 
# [85.17, 90.05, 89.81, 89.78]]

# error = [[0.13, 0.13, 0.14, 0.14], 
# [0.78, 0.16, 0.13, 0.48]]

# one percent true
# one percent false
data = [[89.92, 89.63, 84.90, 89.62, 89.86], 
[90.03, 89.86, 85.02, 89.75, 89.88]]

error =[[0.12, 0.48, 0.16, 0.38, 0.06], 
[0.05, 0.11, 0.20, 0.09, 0.09]]


x = [0.1, 1.1, 2.1, 3.1, 4.1]
# print(len(x))
values = ['all features', 'all features \n except \n jaccard', 
'edit, lcs, \n and vcw, ',  'edit \n and jaccard', 'hamming \n and jaccard']
# print(len(values))

X = np.arange(5)
fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0, 0,1, 1])
ax.bar(X + 0.00, data[0], yerr = error[0], color = 'm', width = 0.25)
ax.bar(X + 0.25, data[1], yerr = error[1], color = 'b', width = 0.25)
# ax.bar(X + 0.5, data[2], yerr = error[2], color = 'b', width = 0.25)
font = {'size'   : 22}
plt.rc('font', **font)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=15)
plt.ylim([84, 91])
plt.xticks(x, values, rotation = 45)
plt.xlabel("Feature Combinations", size = 20)
plt.ylabel("Accuracy", size = 20)
plt.legend(['1% true, 99% false', '1% false, 99% true'], loc= 2)
plt.show()
plt.savefig("results/imbalanced-graph.png", bbox_inches = 'tight')
# %%
