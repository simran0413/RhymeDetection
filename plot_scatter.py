#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/med_half_new.csv')
input = np.asarray(df[['edit_distance', 'hamming_distance', 'jaccard_similarity', 'longest_common_substring', 'weighed_phonemes']])
output = np.asarray(df['output'])
colors = {True: 'red', False: 'blue'}
fig = plt.figure(figsize=(4,4))
plt.xlabel("jaccard")
plt.ylabel("vcw")
# plt.scatter
ax = fig.add_subplot(111)
# for data in df['output']:
#     print(type(data))
# print(df['weighed_phonemes'].max())
ax.scatter(df['jaccard_similarity'], df['weighed_phonemes'], c=df['output'].map(colors), alpha = 0.2)
# ax.scatter(df['edit_distance'], df['hamming_distance'], c=df['output'].map(colors), alpha = 0.2)
# ax.scatter(df['hamming_distance'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
# ax.scatter(df['jaccard_similarity'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
# ax.scatter(df['weighed_phonemes'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
# ax.scatter(df['longest_common_substring'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)

plt.show()
fig.savefig("scatter_jaccard_weighed_half.png")
# %%
