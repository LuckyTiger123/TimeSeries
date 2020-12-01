import pandas as pd
import numpy as np
import las

well_name = '27-141'
level = '61'

raw_data = pd.read_excel(io="/data/luckytiger/shengliOilWell/砂体数据表.xlsx")

item = las.LASReader('/data/luckytiger/shengliOilWell/data/{}'.format(well_name + '.las'))
well_data = pd.DataFrame(item.data)

Top = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == level)]['Top'].head(1))
Bot = float(raw_data[(raw_data['WellName'] == well_name) & (raw_data['XCH'] == level)]['Bot'].head(1))
TopH = Top - 3
BotH = Bot + 3

select_data = well_data[(well_data['DEPTH'] >= TopH) & (well_data['DEPTH'] <= BotH)]
select_data['type'] = select_data['DEPTH'].apply(lambda x: 'in' if Top <= x <= Bot else 'out')

select_data = select_data.replace(-9999, np.nan)

select_data.to_excel("/data/luckytiger/shengliOilWell/InLevel/{}/{}.xlsx".format(level, well_name))
print('Mission complete!')
