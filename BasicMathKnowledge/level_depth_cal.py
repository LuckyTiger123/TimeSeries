import pandas as pd

raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")

level_depth = raw_data.groupby('XCH')['Top', 'Bot'].agg(['max', 'min', 'mean', 'var'])
print(level_depth)
level_depth.to_csv('/data/luckytiger/shengliOilWell/level_depth_cal')
