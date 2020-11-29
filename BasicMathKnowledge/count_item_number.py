import os
import las

item_count = dict()
fail_list = list()
g = os.walk(r"/data/luckytiger/shengliOilWell/data/")
for path, _, fileList in g:
    for file in fileList:
        filePath = os.path.join(path, file)
        try:
            log = las.LASReader(filePath)
            itemCount = len(log.data[0])
            if itemCount in item_count:
                item_count[itemCount].append(file)
            else:
                item_count[itemCount] = [file]
        except ValueError:
            fail_list.append(file)
            print("check {} failed".format(file))
        else:
            print("check {} with item number {}".format(file, itemCount))

# result calculate
f = open(r"/data/luckytiger/shengliOilWell/item_count", "w")
max = 0
maxItem = 0
for key in item_count:
    f.write("item number :{} ,well number: {}\n".format(key, len(item_count[key])))
    if len(item_count[key]) > maxItem:
        maxItem = len(item_count[key])
        max = key
f.close()

f = open(r"/data/luckytiger/shengliOilWell/use_file", "w")
for item in item_count[max]:
    f.write("{}\n".format(item))
f.close()

f = open(r"/data/luckytiger/shengliOilWell/failed_list", "w")
for item in fail_list:
    f.write("{}\n".format(item))
f.close()

print("Mission complete!")
