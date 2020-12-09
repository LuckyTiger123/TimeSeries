import pandas as pd
import numpy as np
import las

raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")
train_set = pd.read_excel(io='/data/luckytiger/shengliOilWell/test_attribute_statistic.xlsx')

use_set = train_set[(train_set['level_cover_rate'] > 0.99) & (train_set['seg_count'] == 1)]
for _, row in use_set.iterrows():
    well_name = row['well_name']
    item = las.LASReader('/data/luckytiger/shengliOilWell/data/{}.las'.format(well_name))
    item_data = pd.DataFrame(item.data)
    use_data = item_data[(item_data['Por'] != -9999) & (item_data['Perm'] != -9999) & (item_data['Por'] != -9999) & (
            item_data['AC'] != -9999) & (item_data['SP'] != -9999) & (
                                 item_data['COND'] != -9999) & (item_data['ML1'] != -9999) & (
                                 item_data['ML2'] != -9999)]
    use_data = use_data.drop(columns=['GR'])
    use_data['level'] = 60
    level_data = raw_data[raw_data['WellName'] == well_name]
    for _, level_row in level_data.iterrows():
        use_data.loc[(use_data['DEPTH'] >= level_row['Top']) & (use_data['DEPTH'] <= level_row['Bot']), 'level'] = \
            level_row['XCH']
    use_data.reset_index(drop=True, inplace=True)
    use_data.to_excel('/data/luckytiger/shengliOilWell/test_data/{}.xlsx'.format(well_name))
    print('well {} rewrite success!'.format(well_name))

print('Mission complete!')
