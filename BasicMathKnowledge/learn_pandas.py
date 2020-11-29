import pandas as pd
import numpy as np
import las

# raw_data = pd.read_excel(io="/data/luckytiger/胜利油田/砂体数据表.xlsx")
# print(raw_data.head())
well_info = ['21-22']

# raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")
# print(raw_data.head())
# use_data = raw_data[raw_data['WellName'].isin(well_info)]

item = las.LASReader('/data/luckytiger/shengliOilWell/data/{}'.format("21-22.las"))
wellInfo = pd.DataFrame(item.data)

indexTop = wellInfo[wellInfo['DEPTH'] >= 1280.6].head(1).index
indexBot = wellInfo[wellInfo['DEPTH'] <= 1283.7].tail(1).index

cross_level_change = pd.DataFrame(
    columns=['WellName', 'XCH', 'type', 'Por', 'Perm', 'AC', 'GR', 'SP', 'COND', 'ML1', 'ML2'])


def calRate(wellInfo, index1, index2):
    result = list()
    for i in range(1, 9):
        if float(wellInfo.iloc[index1, i]) == -9999 or float(wellInfo.iloc[index2, i]) == -9999:
            result.append(np.nan)
        else:
            itemR = (float(wellInfo.iloc[index1, i]) - float(wellInfo.iloc[index2, i])) / float(
                wellInfo.iloc[index2, i])
            result.append(itemR)
    return result


a = list()
a.extend(["21-22", "61", 'Top'])
item = calRate(wellInfo, indexTop, indexTop - 1)
a.extend(item)
cross_level_change.loc[cross_level_change.shape[0]] = a
cross_level_change.to_excel('/data/luckytiger/shengliOilWell/cross_level.xlsx')
