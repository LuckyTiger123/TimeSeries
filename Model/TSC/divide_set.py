from random import sample
import os

for i in range(1, 9):
    g = os.walk('/data/luckytiger/shengliOilWell/TSC/train_data/level{}'.format(i))
    for _, _, file_list in g:
        loc_len = int(len(file_list) * 0.9)
        loc_set = sample(file_list, loc_len)
        adjust_set = list(set(file_list) - set(loc_set))
        f = open('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/loc_set'.format(i), "w")
        for item in loc_set:
            f.write("{}\n".format(item))
        f.close()

        f = open('/data/luckytiger/shengliOilWell/TSC/train_data/level{}/adjust_set'.format(i), "w")
        for item in adjust_set:
            f.write("{}\n".format(item))
        f.close()

        print('level {} has been divided.'.format(i))

print('mission complete!')
