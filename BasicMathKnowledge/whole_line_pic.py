import seaborn as sns
import pandas as pd

well_name = '21-22'

data = pd.read_excel("/data/luckytiger/shengliOilWell/WholeLine/{}.xlsx".format(well_name))
sns.relplot(x='DEPTH', y='Por', data=data, hue='type', aspect=3)

well_name = '35-194'

data = pd.read_excel("/data/luckytiger/shengliOilWell/WholeLine/{}.xlsx".format(well_name))
sns.relplot(x='DEPTH', y='Por', data=data, hue='type', aspect=3)

well_name = '27-141'

data = pd.read_excel("/data/luckytiger/shengliOilWell/WholeLine/{}.xlsx".format(well_name))
sns.relplot(x='DEPTH', y='Por', data=data, hue='type', aspect=3)
