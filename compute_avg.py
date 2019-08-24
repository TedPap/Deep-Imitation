import csv
import os
import numpy as np

def input_data(path):

    i = 0
    data = {}
    sumCon = [0 for i in range(100)]
    sumObs = [0 for i in range(100)]
    steps = [x for x in range(10,1010,10)]
    for file in sorted(os.listdir(path)):
        print(file)
        filename = path+'/'+file
        with open(filename, newline='') as csvfile:
            filereader = csv.reader(csvfile)
            data[i] = np.float32([r for r in filereader])

        i+=1
    print(len(data))
    for index in range(20):
        tmp = data[index]
        for i in range(len(tmp)):
            sumCon[i] += tmp[i][1]
    sumCon = [ cell/20 for cell in sumCon]
    print(sumCon)

    for index in range(20,40):
        tmp2 = data[index]
        print(len(tmp2))
        for i in range(len(tmp2)):
            sumObs[i] += tmp2[i][1]
    sumObs = [ cell/20 for cell in sumObs]
    print(sumObs)
    print(steps)


    with open("logAvgContObs.csv",  mode='a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(steps)):
            filewriter.writerow([steps[i], sumCon[i], sumObs[i]])


input_data("logs/mountaincar")