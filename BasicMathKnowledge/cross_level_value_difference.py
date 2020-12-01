import pandas as pd
import numpy as np
import las

well_info = dict()

raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")
file = open("/data/luckytiger/shengliOilWell/use_file", 'r')
fileName = file.readline()
while fileName:
    fileName = fileName.replace("\n", "")
    item = las.LASReader('/data/luckytiger/shengliOilWell/data/{}'.format(fileName))
    well_info[fileName.replace(".las", "")] = pd.DataFrame(item.data)
    print("read {} success!".format(fileName.replace(".las", "")))
    fileName = file.readline()

use_data = raw_data[raw_data['WellName'].isin(well_info.keys())]

cross_level_change = pd.DataFrame(
    columns=['WellName', 'XCH', 'type', 'Por', 'Perm', 'AC', 'GR', 'SP', 'COND', 'ML1', 'ML2'])


def calRate(wellInfo, index1, index2):
    result = list()
    for i in range(1, 9):
        if float(wellInfo.iloc[index1, i]) == -9999 or float(wellInfo.iloc[index2, i]) == -9999:
            result.append(np.nan)
        else:
            if float(wellInfo.iloc[index2, i]) == 0:
                result.append(np.nan)
                continue
            itemR = float(wellInfo.iloc[index1, i]) - float(wellInfo.iloc[index2, i])
            result.append(itemR)
    return result


for index, row in use_data.iterrows():
    rowTopResult = list()
    rowBotResult = list()
    check_data = well_info[row['WellName']]
    indexTop = check_data[check_data['DEPTH'] >= row['Top']].head(1).index
    indexBot = check_data[check_data['DEPTH'] <= row['Bot']].tail(1).index
    itemTopResult = calRate(check_data, indexTop, indexTop - 1)
    itemBotResult = calRate(check_data, indexBot + 1, indexBot)

    rowTopResult.extend([str(row['WellName']), str(row['XCH']), 'Top'])
    rowTopResult.extend(itemTopResult)

    rowBotResult.extend([str(row['WellName']), str(row['XCH']), 'Bot'])
    rowBotResult.extend(itemBotResult)

    print(rowTopResult)
    print(rowBotResult)

    cross_level_change.loc[cross_level_change.shape[0]] = rowTopResult
    cross_level_change.loc[cross_level_change.shape[0]] = rowBotResult

cross_level_change.to_excel('/data/luckytiger/shengliOilWell/cross_level_value_difference.xlsx')
print('Mission complete!')
