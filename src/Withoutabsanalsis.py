import os
import pandas as pd
from Crowdproportion import *
from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns; sns.set(style="white", color_codes=True)


current_dir =  os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../")
print parent_dir

label=dict()
lengthlabel=dict()
dataset = pd.read_csv(os.path.join(os.path.abspath(current_dir + "/../"), 'data', 'ProtonBeamComplete.txt'), header=0,
                    delimiter='\t', quoting=3)

for i in xrange(0, len(dataset["abstract"])):

    lengthlabel[str(dataset["pid"][i])]=[len(dataset["abstract"][i].split()),dataset["label"][i]]
    if(dataset["abstract"][i] == '\N'):
        label[str(dataset["pid"][i])] = dataset["label"][i]
        lengthlabel[str(dataset["pid"][i])]=[0,dataset["label"][i]]






d=CrowsDis(filename='../data/ProtonBeamCrowddata.txt')
d.proportion()
d.voting()
ans=d.getvotes()
ansfirstquestion=d.getcrowdvotequestion1()









crowd={k:v for (k,v) in ans.items() if v==1}
Expert={k:v for (k,v) in lengthlabel.items() if v[1]==1}
fcrowd={k:v for (k,v) in ansfirstquestion.items() if v==1}
print len(fcrowd)
print len(crowd)
print len(Expert)

with open('../data/NotAbstract.csv','w') as f:
    for item in label:

        f.write(item+','+str(label[item])+','+str(ans[item])+'\n')





with open('../data/lenlabelCorrelation.csv','w') as f:
    for item in lengthlabel:

        if(int(lengthlabel[item][1])==int(ans[item])):
            f.write(str(item)+","+str(lengthlabel[item][0])+','+str(int(lengthlabel[item][1]))+','+str(int(ans[item]))+',1'+'\n')
        else:
            f.write(str(item) + "," + str(lengthlabel[item][0]) + ',' + str(int(lengthlabel[item][1])) + ',' + str(int(ans[item])) + ',0' + '\n')






#plot distribution

correlationdata = pd.read_csv(os.path.join(os.path.abspath(current_dir + "/../"), 'data', 'lenlabelCorrelation.csv'), header=None,
                              sep=',', quoting=3)

X = correlationdata[[1]].values
y = correlationdata[4].values
print len(X[y == 1, 0])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
label_dict = {0: 'NotAgree', 1: 'Agree'}
color={0:'red',1:'blue'}
distribution=defaultdict(list)
for ax,lab in zip(axes.ravel(),range(0,2)):
    min_b = math.floor(np.min(X[:,0]))
    max_b = math.ceil(np.max(X[:,0]))
    bins = np.linspace(min_b, max_b, 25)

    n=ax.hist(X[y == lab, 0],
            color=color[lab],
            label='class %s' % label_dict[lab],
            bins=bins,
            alpha=0.5)

    distribution[lab]=n[0]
    ylims = ax.get_ylim()

    # plot annotation
    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims)+2])
    ax.set_xlabel("length")
    ax.set_title('length versus'+label_dict[lab])

    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


# axes[0][0].set_ylabel('count')

fig.tight_layout()

plt.show()
print len(distribution)
for item in distribution:
    print distribution[item]


result=[]
for item in range(0,24):
    result.append(distribution[1][item]/(distribution[0][item] + distribution[1][item]))

min_b = math.floor(np.min(X[:, 0]))
max_b = math.ceil(np.max(X[:, 0]))
bins = np.linspace(min_b, max_b, 25)
bins=bins[:24]
plt.plot(bins,result,'ro')
plt.xlabel("length")
plt.ylabel("Correctness probability")
plt.ylim(0,1.1)
plt.show()


for lab,col in zip(range(0,2), ('green', 'blue')):
    plt.hist(X[y == lab, 0],
             color=col,
             label='class %s' % label_dict[lab],
             bins=bins,
             alpha=0.5)


leg = plt.legend(loc='upper right', fancybox=True, fontsize=8)
leg.get_frame().set_alpha(0.5)
plt.xlabel("length")
plt.title('length versus'+label_dict[lab])

plt.tight_layout()
plt.show()