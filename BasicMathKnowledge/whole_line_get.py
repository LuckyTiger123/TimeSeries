import pandas as pd
import numpy as np
import las

well_name = '27-141'

raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")

item = las.LASReader('/data/luckytiger/shengliOilWell/data/{}'.format(well_name + '.las'))
well_data = pd.DataFrame(item.data)

level_list = list()

checkTop = list()
checkBot = list()

dataInfo = raw_data[raw_data['WellName'] == well_name]
for index, row in dataInfo.iterrows():
    item = list()
    item.append(float(row['Top']))
    item.append(float(row['Bot']))
    item.append(row['XCH'])
    level_list.append(item)
    checkTop.append(item[0])
    checkBot.append(item[1])

TopM = min(checkTop) - 3
BotM = max(checkBot) + 3

select_data = well_data[(well_data['DEPTH'] >= TopM) & (well_data['DEPTH'] <= BotM)]

select_data['type'] = 'out'
for i in level_list:
    select_data.loc[(select_data['DEPTH'] >= i[0]) & (select_data['DEPTH'] <= i[1]), 'type'] = i[2]

# try:
#     Top1 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '61')]['Top'].head(1))
#     Bot1 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '61')]['Bot'].head(1))
# except TypeError:
#     Top1 = 0
#     Bot1 = 0
# else:
#     checkTop.append(Top1)
#     checkBot.append(Bot1)
#
# try:
#     Top2 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '62')]['Top'].head(1))
#     Bot2 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '62')]['Bot'].head(1))
# except TypeError:
#     Top2 = 0
#     Bot2 = 0
# else:
#     checkTop.append(Top2)
#     checkBot.append(Bot2)
#
# try:
#     Top3 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '63')]['Top'].head(1))
#     Bot3 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '63')]['Bot'].head(1))
# except TypeError:
#     Top3 = 0
#     Bot3 = 0
# else:
#     checkTop.append(Top3)
#     checkBot.append(Bot3)
#
# try:
#     Top4 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '64')]['Top'].head(1))
#     Bot4 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '64')]['Bot'].head(1))
# except TypeError:
#     Top4 = 0
#     Bot4 = 0
# else:
#     checkTop.append(Top4)
#     checkBot.append(Bot4)
#
# try:
#     Top5 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '65')]['Top'].head(1))
#     Bot5 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '65')]['Bot'].head(1))
# except TypeError:
#     Top5 = 0
#     Bot5 = 0
# else:
#     checkTop.append(Top5)
#     checkBot.append(Bot5)
#
# try:
#     Top6 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '66')]['Top'].head(1))
#     Bot6 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '66')]['Bot'].head(1))
# except TypeError:
#     Top6 = 0
#     Bot6 = 0
# else:
#     checkTop.append(Top6)
#     checkBot.append(Bot6)
#
# try:
#     Top7 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '67')]['Top'].head(1))
#     Bot7 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '67')]['Bot'].head(1))
# except TypeError:
#     Top7 = 0
#     Bot7 = 0
# else:
#     checkTop.append(Top7)
#     checkBot.append(Bot7)
#
# try:
#     Top8 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '68')]['Top'].head(1))
#     Bot8 = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == '68')]['Bot'].head(1))
# except TypeError:
#     Top8 = 0
#     Bot8 = 0
# else:
#     checkTop.append(Top8)
#     checkBot.append(Bot8)
#
# TopM = min(checkTop) - 3
# BotM = max(checkBot) + 3
#
# select_data = well_data[(well_data['DEPTH'] >= TopM) & (well_data['DEPTH'] <= BotM)]
#
# select_data['type'] = 'out'
# select_data.loc[(select_data['DEPTH'] >= Top1) & (select_data['DEPTH'] <= Bot1), 'type'] = '61'
# select_data.loc[(select_data['DEPTH'] >= Top2) & (select_data['DEPTH'] <= Bot2), 'type'] = '62'
# select_data.loc[(select_data['DEPTH'] >= Top3) & (select_data['DEPTH'] <= Bot3), 'type'] = '63'
# select_data.loc[(select_data['DEPTH'] >= Top4) & (select_data['DEPTH'] <= Bot4), 'type'] = '64'
# select_data.loc[(select_data['DEPTH'] >= Top5) & (select_data['DEPTH'] <= Bot5), 'type'] = '65'
# select_data.loc[(select_data['DEPTH'] >= Top6) & (select_data['DEPTH'] <= Bot6), 'type'] = '66'
# select_data.loc[(select_data['DEPTH'] >= Top7) & (select_data['DEPTH'] <= Bot7), 'type'] = '67'
# select_data.loc[(select_data['DEPTH'] >= Top8) & (select_data['DEPTH'] <= Bot8), 'type'] = '68'
#
select_data = select_data.replace(-9999, np.nan)

select_data.to_excel('/data/luckytiger/shengliOilWell/WholeLine/{}.xlsx'.format(well_name))
print('Mission complete!')
