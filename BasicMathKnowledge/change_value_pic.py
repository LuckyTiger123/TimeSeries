import seaborn as sns
import pandas as pd

value_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/cross_level_value_difference.xlsx")

use_data = value_data[value_data['type'] == 'Top']
sns.stripplot(x='XCH', y='Por', data=use_data, jitter=0.3)

use_data = value_data[value_data['type'] == 'Bot']
sns.stripplot(x='XCH', y='Por', data=use_data, jitter=0.3)

use_data = value_data[(value_data['type'] == 'Top') & (value_data['Perm'] < 2000) & (value_data['Perm'] > -2000)]
sns.stripplot(x='XCH', y='Perm', data=use_data, jitter=0.3)

use_data = value_data[(value_data['type'] == 'Bot') & (value_data['Perm'] < 2000) & (value_data['Perm'] > -2000)]
sns.stripplot(x='XCH', y='Perm', data=use_data, jitter=0.3)

use_data = value_data[(value_data['type'] == 'Top') & (value_data['AC'] < 50) & (value_data['AC'] > -50)]
sns.stripplot(x='XCH', y='AC', data=use_data, jitter=0.3)

use_data = value_data[(value_data['type'] == 'Bot')]
sns.displot(use_data, x='Por', kind='kde', col='XCH')

use_data = value_data[(value_data['type'] == 'Top') & (value_data['Perm'] < 2000) & (value_data['Perm'] > -2000)]
sns.displot(use_data, x='Perm', kind='kde', col='XCH')

use_data = value_data[(value_data['type'] == 'Top')]
sns.displot(use_data, x='AC', kind='kde', col='XCH')
