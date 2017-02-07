import unicodedata
import string
import numpy as np


def createSingle29(input, currentLetterIndex, current29Index):

    go = 1
    finalV =[]
    for i in range(0,30):
        finalV.append(0)
    if 1+currentLetterIndex+current29Index > len(input):
        finalV[29]=1
    elif input[currentLetterIndex + current29Index].isalpha():
        letter = input[currentLetterIndex + current29Index]
        finalV[string.ascii_lowercase.index(letter)] = 1
    elif input[currentLetterIndex + current29Index] == "-":
        finalV[26] = 1
    elif input[currentLetterIndex + current29Index] == "'":
        finalV[27] = 1
    elif input[currentLetterIndex + current29Index] == " ":
        finalV[28] = 1
    else:
        go = 0
    finalV.append(len(input))

    return go, finalV

print(createSingle29("abc",0,1))

def createsFeaturesForLetter(inputStr,indexLetter):
    go = 1
    vecFeatures = []
    vecLabel = []
    for i in range(0,30):
        vecLabel.append(0)
    if inputStr[indexLetter].isalpha():
        letter = inputStr[indexLetter]
        vecLabel[string.ascii_lowercase.index(letter)] = 1
    elif inputStr[indexLetter]== "-":
        vecLabel[26] = 1
    elif inputStr[indexLetter]== "'":
        vecLabel[27] = 1
    elif inputStr[indexLetter]== " ":
        vecLabel[28] = 1
    else:
        go = 0

    for ind29 in range(1, 21):
        go, vec = createSingle29(inputStr, indexLetter, ind29)
        vecFeatures.extend(vec)

    #vecFeatures.append(len(inputStr))

    return go, vecFeatures, vecLabel


def createsFeaturesForString(str):
    str = str.lower()

    finalData = []
    for i in range(0,len(str)):
        go, vecFeatures, vecLabel = createsFeaturesForLetter(str, i)
        data = []
        data.append(vecFeatures)
        data.append(vecLabel)
        finalData.append(data)

    finalData = np.array(finalData)
    features_x = finalData[:, 0]
    features_y = finalData[:, 1]
    return features_x, features_y


x1, y1 = createsFeaturesForString("coucou")
print(len(x1[0]))
# x2, y2 = createsFeaturesForString("salut")
#
# data = []
#
# data = np.append(data,x1)
# data = np.append(data,x2)
# print(data)
# print("Length",len(data))
# print("Length1",len(data[0]))


# data = createsFeaturesForString("Coucou".lower())
# data = np.array(data)
# a = list(data[:,1])
# print(a)

