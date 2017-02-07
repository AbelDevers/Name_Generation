import numpy as np
import pickle
import featuresFunction as ff
import random


testing_size = 1000

with open("C:/Users/Antoine Didisheim/Dropbox/PyCharm/TensorFlowTutorials/Geodata/GEODATASOURCE-SUBREGION.TXT", "r") as ins:
    subregionName = []
    for line in ins:
        subregionName.append(line[0:2])
i = 0
cc_fips = 'CH'
print("loading countries from: ",cc_fips)

with open("C:/Users/Antoine Didisheim/Dropbox/PyCharm/TensorFlowTutorials/Geodata/GEODATASOURCE-CITIES-FREE.TXT", "r",encoding="utf8") as cityList:
    cityName = []
    for line in cityList:
        if line[0:2] == cc_fips:
            cityName.append(line[3:-1])
print("loading finished")

random.shuffle(cityName)
print(cityName)
print("Length of list", len(cityName))

print("Starting featurization process")
features_x = []
features_y = []
test_x = []
test_y = []

for name in cityName:
    try:
        x, y = ff.createsFeaturesForString(name)
        x = np.array(x)
        y = np.array(y)
        features_x = np.append(features_x, x)
        features_y = np.append(features_y, y)

        for v in x:
            if len(v)!=620:
                print("error X", name,len(v))
        for v in y:
            if len(v) != 30:
             print("error Y", name,len(v))
    except:
        print("Some error with word: ",name)

train_x = list(features_x[:testing_size])
train_y = list(features_y[:testing_size])

for v in train_x:
    if len(v) != 620:
        print("error X", len(v))
for v in train_y:
    if len(v) != 30:
        print("error Y", len(v))

test_x = list(features_x[testing_size:])
test_y = list(features_y[testing_size:])


trainData = [train_x, train_y]

pickle.dump(trainData, open("train_"+cc_fips+".p", "wb"))

testData = [test_x, test_y]
pickle.dump(testData, open("test_"+cc_fips+".p", "wb"))




