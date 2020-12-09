import seaborn as sns
import pandas as pd

data = pd.read_excel('/data/luckytiger/shengliOilWell/attribute_statistic.xlsx')

sns.displot(data, x='seg_count')

sns.displot(data, x='level_cover_rate', kind='kde')
