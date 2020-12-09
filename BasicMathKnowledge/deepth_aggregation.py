import seaborn as sns
import pandas as pd

raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")
group_data = raw_data.groupby('WellName').agg({'Top': lambda x: x.min(),
                                               'Bot': lambda x: x.max()})
group_data['Length'] = group_data['Bot'] - group_data['Top']
sns.displot(data=group_data, x='Length')

# sns.relplot(x='XCH', y='Top', data=raw_data)
#
# raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")
# raw_data['Length'] = None
# for i in range(raw_data.shape[0]):
#     raw_data.loc[i, 'Length'] = raw_data.loc[i, 'Bot'] - raw_data.loc[i, 'Top']
# raw_data = raw_data[raw_data['Length'] < 25]
# sns.relplot(x='XCH', y='Length', data=raw_data, aspect=0.5, height=10)
