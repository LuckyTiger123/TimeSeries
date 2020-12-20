import pandas as pd
import las

train_well_info = dict()
level_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")

file = open("/data/luckytiger/shengliOilWell/train_set", 'r')
fileName = file.readline()
while fileName:
    fileName = fileName.replace("\n", "")
    item = las.LASReader('/data/luckytiger/shengliOilWell/data/{}'.format(fileName))
    train_well_info[fileName.replace(".las", "")] = pd.DataFrame(item.data)
    print("read {} success!".format(fileName.replace(".las", "")))
    fileName = file.readline()
file.close()

for key in train_well_info.keys():
    well_levels = level_data[level_data['WellName'] == key]
    for _, row in well_levels.iterrows():
        level = int(row['XCH']) - 60
        level_Top = row['Top']
        level_Bot = row['Bot']
        TS_item = train_well_info[key]
        TS_item = TS_item[(TS_item['DEPTH'] >= level_Top) & (TS_item['DEPTH'] <= level_Bot)]
        TS_item.reset_index(drop=True, inplace=True)
        TS_item.drop(columns=['GR'], inplace=True)
        TS_item.to_excel('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/{}.xlsx'.format(level, key))
        print('Train well {} \'s level {} has been divided'.format(key, level))

test_well_info = dict()

file = open("/data/luckytiger/shengliOilWell/test_set", 'r')
fileName = file.readline()
while fileName:
    fileName = fileName.replace("\n", "")
    item = las.LASReader('/data/luckytiger/shengliOilWell/data/{}'.format(fileName))
    test_well_info[fileName.replace(".las", "")] = pd.DataFrame(item.data)
    print("read {} success!".format(fileName.replace(".las", "")))
    fileName = file.readline()
file.close()

for key in test_well_info.keys():
    well_levels = level_data[level_data['WellName'] == key]
    for _, row in well_levels.iterrows():
        level = int(row['XCH']) - 60
        level_Top = row['Top']
        level_Bot = row['Bot']
        TS_item = test_well_info[key]
        TS_item = TS_item[(TS_item['DEPTH'] >= level_Top) & (TS_item['DEPTH'] <= level_Bot)]
        TS_item.reset_index(drop=True, inplace=True)
        TS_item.drop(columns=['GR'], inplace=True)
        TS_item.to_excel('/data/luckytiger/shengliOilWell/TSC/test_data/level{}/{}.xlsx'.format(level, key))
        print('Test well {} \'s level {} has been divided'.format(key, level))

print('Mission complete!')
