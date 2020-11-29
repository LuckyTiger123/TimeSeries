import pandas as pd

raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/cross_level.xlsx")
data = raw_data.groupby(['XCH', 'type'])['Por', 'Perm', 'AC', 'GR', 'SP', 'COND', 'ML1', 'ML2'].agg(
    ['max', 'min', 'mean', 'var'])
data.to_excel('/data/luckytiger/shengliOilWell/aggregate_cross_level.xlsx')
print('mission complete!')
