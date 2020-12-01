import seaborn as sns
import pandas as pd

raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/cross_level.xlsx")
use_data = raw_data[raw_data['type'] == 'Top']
sns.stripplot(x='XCH', y='Por', data=use_data, jitter=0.3)

use_data = raw_data[(raw_data['type'] == 'Bot') & (raw_data['Por'] < 5)]
sns.stripplot(x='XCH', y='Por', data=use_data, jitter=0.3)

use_data = raw_data[(raw_data['type'] == 'Top') & (raw_data['Perm'] < 20000)]
sns.stripplot(x='XCH', y='Perm', data=use_data, jitter=0.3)

use_data = raw_data[(raw_data['type'] == 'Bot') & (raw_data['Perm'] < 2500)]
sns.stripplot(x='XCH', y='Perm', data=use_data, jitter=0.3)

use_data = raw_data[raw_data['type'] == 'Top']
sns.displot(use_data, x='Perm', kind='kde', col='XCH')

use_data = raw_data[raw_data['type'] == 'Top']
sns.displot(use_data, x='AC', kind='kde', col='XCH')

use_data = raw_data[(raw_data['type'] == 'Bot') & (raw_data['Perm'] < 2500)]
sns.displot(use_data, x='Perm', kind='kde', col='XCH')