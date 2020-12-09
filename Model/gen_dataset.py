import pandas as pd
import numpy as np
import las

well_info = dict()

level_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")
statistic_form = pd.DataFrame(
    columns=['well_name', 'available_depth', 'cover_depth', 'whole_depth', 'level_cover_rate', 'proportion_rate',
             'seg_count'])
file = open("/data/luckytiger/shengliOilWell/test_set", 'r')
fileName = file.readline()
while fileName:
    fileName = fileName.replace("\n", "")
    item = las.LASReader('/data/luckytiger/shengliOilWell/data/{}'.format(fileName))
    well_info[fileName.replace(".las", "")] = pd.DataFrame(item.data)
    print("read {} success!".format(fileName.replace(".las", "")))
    fileName = file.readline()

# i = las.LASReader('/data/luckytiger/shengliOilWell/data/{}'.format('21-22.las'))
# well_info['21-22'] = pd.DataFrame(i.data)

for key in well_info.keys():
    raw_data = well_info[key]
    item = raw_data[(raw_data['Por'] != -9999) & (raw_data['Perm'] != -9999) & (raw_data['Por'] != -9999) & (
            raw_data['AC'] != -9999) & (raw_data['SP'] != -9999) & (
                            raw_data['COND'] != -9999) & (raw_data['ML1'] != -9999) & (
                            raw_data['ML2'] != -9999)]

    total_count = item.shape[0]
    levelTop = level_data[level_data['WellName'] == key].head(1)
    if levelTop.shape[0] == 0:
        continue
    level_Top = float(levelTop['Top'])

    levelBot = level_data[level_data['WellName'] == key].tail(1)
    level_Bot = float(levelBot['Bot'])

    cover_count = item[(item['DEPTH'] < level_Bot) & (item['DEPTH'] > level_Top)].shape[0]
    whole_count = int((level_Bot - level_Top) / 0.125)

    if total_count == 0:
        level_c_rate = 0
        level_p_rate = 0
    else:
        level_c_rate = float(cover_count) / float(whole_count)
        level_p_rate = float(cover_count) / float(total_count)

    item.reset_index(drop=True, inplace=True)
    print(item.head())
    seg_count = 1
    for i in range(1, item.shape[0]):
        if item.loc[i, 'DEPTH'] != (item.loc[i - 1, 'DEPTH'] + 0.125):
            seg_count += 1

    statistic_form.loc[statistic_form.shape[0]] = {'well_name': key, 'available_depth': total_count,
                                                   'cover_depth': cover_count, 'whole_depth': whole_count,
                                                   'level_cover_rate': level_c_rate, 'proportion_rate': level_p_rate,
                                                   'seg_count': seg_count}
    print(statistic_form.tail(1))

statistic_form.to_excel('/data/luckytiger/shengliOilWell/test_attribute_statistic.xlsx')

print('Mission complete!')
