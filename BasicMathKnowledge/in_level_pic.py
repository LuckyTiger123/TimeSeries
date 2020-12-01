import seaborn as sns
import pandas as pd

well_name = '21-22'
level = '61'

data = pd.read_excel("/data/luckytiger/shengliOilWell/InLevel/{}/{}.xlsx".format(level, well_name))
sns.relplot(x='DEPTH', y='Perm', data=data, hue='type')
