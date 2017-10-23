#this python file find label for dataset and write it in file label.csv

from collections import Counter
import os
import pandas as pd

current_dir =  os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../")

with open('../data/WIOProtonbeam.csv') as f:
    lines = f.read().splitlines()

whole=[]
Inout=[]
Infinal=[]
for line in lines:
    item=line.split(',')
    whole.append(item[0])
    Inout.append(item[1])
    Infinal.append(item[1])
    Inout.append(item[2])


ff= {v for v in whole if v in Inout}
rr={v for v in whole if v not in Inout}

IF={v for v in whole if v in Infinal}
OF={v for v in Inout if v not in Infinal}
print len(whole)
print len(ff)
print len(rr)
print len(IF)
print len(OF)

label=[]
dataset = pd.read_csv(os.path.join(os.path.abspath(current_dir + "/../"), 'data', 'ProtonBeamComplete.txt'), header=0,
                    delimiter='\t', quoting=3)

for i in xrange(0, len(dataset["abstract"])):

    if str(dataset["pid"][i]) in ff:
        # print dataset["pid"][i]
        label.append("1")
    elif str(dataset["pid"][i]) in rr:

        label.append("0")
    else:
        print dataset["pid"][i]
print len(label)
with open('../data/label.csv','w') as f:
    for item in label:

        f.write(item+'\n')


