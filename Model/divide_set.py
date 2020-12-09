from random import sample

file_list = list()

file = open("/data/luckytiger/shengliOilWell/use_file", 'r')
fileName = file.readline()
while fileName:
    fileName = fileName.replace("\n", "")
    file_list.append(fileName)
    fileName = file.readline()

train_set = sample(file_list, 550)
test_set = list(set(file_list) - set(train_set))

f = open(r"/data/luckytiger/shengliOilWell/train_set", "w")
for item in train_set:
    f.write("{}\n".format(item))
f.close()

f = open(r"/data/luckytiger/shengliOilWell/test_set", "w")
for item in test_set:
    f.write("{}\n".format(item))
f.close()


