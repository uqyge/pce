#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
df_nodes_1 = pd.read_csv('./data/gauss1_samples(1).csv')
df_nodes_1.describe()

# %%
df_nodes_2 = pd.read_csv('./data/gauss2_samples(1).csv')
df_nodes_2.describe()

# %%
df_mlab_1 = pd.read_csv('./data/displace_23.csv')
df_mlab_1.describe()
# %%
df_mlab_2 = pd.read_csv('./data/displace_23_2.csv')
df_mlab_2.describe()
# %%

# %%
df_mlab_2['Z23'].hist(bins=20)
# %%
df_mlab_1['Z23'].hist(bins=20)
# %%
df_nodes_1['1'].value_counts()
# %%
df_nodes_2['5'].value_counts()
# %%
3**7
# %%
5**7
# %%
df_nodes_1[['0','1','2','3']].unique
# %%
df_err = df_mlab_2[df_mlab_2['Z23']<0]
# %%
cond =(df_mlab_1['drive_2']>df_mlab_1['drive_1'])
# %%
(df_mlab_2['drive_2']>df_mlab_2['drive_1']).sum()
# %%
df_mlab_1[cond]
# %%
plt.scatter(df_err['drive_1'],df_err['drive_2'])
# %%
df_nodes_1

# %%
df_rev = df_nodes_1.iloc[::-1]
# %%
df_rev.to_csv('gauss1_reversed.csv')
# %%
2**7
# %%
3**7
# %%
3**3
# %%
