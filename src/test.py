import pandas as pd
import os


current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../")

Wdataset = pd.read_csv(os.path.join(os.path.abspath(current_dir + "/../"), 'data', 'proton-beam-merged.txt'), header=0,
                    delimiter='\t', quoting=3)




dataset = pd.read_csv(os.path.join(os.path.abspath(current_dir + "/../"), 'data', 'ProtonBeamComplete.txt'), header=0,
                    delimiter='\t', quoting=3)

wdic={}
for item in range(0,len(Wdataset)):

    wdic[Wdataset['pmid'][item]] = 0 if Wdataset['label'][item] == -1 else Wdataset['label'][item]


with open('wlabel','w') as f:

    for item in range(0,len(dataset)):
        t=wdic[dataset['pid'][item]]
        f.write(str(t)+'\n')








