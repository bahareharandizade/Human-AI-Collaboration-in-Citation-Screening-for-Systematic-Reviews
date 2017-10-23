import os
import pandas as pd
import random
from scipy.stats import bernoulli
import mainBM25
import numpy as np
import pickle
from Crowdproportion import *
from Crowd import *
from Clusters import clusterDocs



class InitialExperimant:
    def __init__(self,mode,k):
        self.k=k
        self.mode=mode
        self.learningset=pd.DataFrame()
        self.restset=pd.DataFrame()
        self.orginallearningset=pd.DataFrame()
        self.orginalrestingset=pd.DataFrame()
        current_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.abspath(current_dir + "/../")

        self.dataset = pd.read_csv(os.path.join(os.path.abspath(current_dir + "/../"), 'data', 'proton-beam-merged.txt'), header=0,
                    delimiter='\t', quoting=3)
        self.feature_dict = {i: label for i, label in zip(
            range(3),
            ('pmid',
             'text',
             'label'))}
        self.dataset['pmid'] = self.dataset['pmid'].astype(int).astype(str)

        self.dataset = askCrowd(self.dataset, '1')




    def Random(self):

        if  self.mode =='1':
            randdata=(np.array(random.sample(self.dataset[[0,1]].values,self.k))).tolist()
            randlabel=(np.array([bernoulli.rvs(.5,size=self.k)]).T).tolist()


        elif self.mode =='2':
            randdata=(self.dataset[[0,1]].values).tolist()
            randlabel= (np.array([bernoulli.rvs(.5,size=len(self.dataset["pmid"]))]).T).tolist()

        data = np.array([t1 + t2 for t1, t2 in zip(randdata, randlabel)])
        columns=[l for i, l in sorted(self.feature_dict.items())]
        self.learningset=pd.DataFrame(data,columns=columns)
        self.dataset['pmid'] = self.dataset['pid'].astype(int).astype(str)
        self.restset = self.dataset[~self.dataset['pid'].isin(self.learningset['pmid'])]
        #put here comment if you don't want to use question 1 as GT
        self.restset=askCrowd(self.restset,'1')#change the label to MV question 1
        self.restset['pmid'] = self.restset['label'].astype(str).astype(int)
        print self.restset[[5]].values

        self.orginallearningset=self.learningset
        self.orginalrestingset=self.restset


    def Balanced(self):
        #from dataset find 10 + and 10 -
        print self.dataset
        indataset = self.dataset.loc[self.dataset['label'] == '1']
        outdataset = self.dataset.loc[self.dataset['label'] == '0']
        print len(indataset)
        randIndata = np.array(random.sample(indataset[[0, 1, 2]].values, self.k/2))
        randOutdata = np.array(random.sample(outdataset[[0, 1, 2]].values, self.k / 2))
        data = np.concatenate((randIndata,randOutdata),axis=0)

        columns = [l for i, l in sorted(self.feature_dict.items())]
        self.learningset = pd.DataFrame(data, columns=columns)

        self.restset = self.dataset[~self.dataset['pmid'].isin(self.learningset['pmid'])]


        self.orginallearningset = self.learningset
        self.orginalrestingset = self.restset
    def BM25(self,k1,k2,data):

        rankeddataset, score = mainBM25.BM25(data)
        # since the distribution is power-low we use 80-20 policy for assign + and -
        threshold = 0.2 * len(rankeddataset)
        randlabel = (np.array([np.array([0] * threshold + [1] * (len(rankeddataset["pid"]) - threshold))]).T).tolist()
        randdata = (self.rankeddataset[[0, 1, 2, 3, 4]].values).tolist()
        data = np.array([t1 + t2 for t1, t2 in zip(randdata, randlabel)])
        columns = [l for i, l in sorted(self.feature_dict.items())]


        if self.mode=='1':
            #0.6 positive and 0.4 negative
            self.learningset = pd.DataFrame(np.concatenate((data[0:(k1*self.k),:],data[-1:(k2*self.k):-1]),axis=0),columns=columns)
            self.restset = self.dataset[~self.dataset['pid'].isin(self.learningset['pid'])]

        if self.mode=='2':
            self.learningset = pd.DataFrame(data, columns=columns)


    def Clustering(self):
        if os.path.exists('../data/clusters.pkl')==False:
            clus=clusterDocs()
            results = []
            data=self.dataset[[2]].values
            for cluster in ['kmean']:
                results[cluster] = dict()
                for i in range(2, 11):
                    km, samples = clus.Kmeans(i, data, self.k)  # i number of clusters
                    results[cluster][i] = samples,km

            pickle.dump(results, open('../data/clusters.pkl', 'w'))


        else:
            data = pickle.load(open('../data/clusters.pkl', 'r'))
            #example run
            samples=data['kmean'][2][0]
            clusterdataframe = pd.DataFrame()
            clusterdataframe.columns = [l for i, l in sorted(self.feature_dict.items())]
            for i in samples:
                item=self.dataset['pid'][i]
                row = self.dataset.loc[self.dataset['pid'] == item]
                clusterdataframe.append(row, ignore_index=True)

            return clusterdataframe


    def crowd(self,seleteddata):

        crowddata = (self.seleteddata[[0, 1, 2, 3, 4]].values).tolist()
        d = CrowsDis(filename='../data/ProtonBeamCrowddata.txt')
        d.proportion()
        d.voting()
        ans = d.getvotes()
        votes=[]
        for item in range(0,len(seleteddata["pid"])):
            votes.append(ans[seleteddata["pid"][item]])

        crowdlabel = (np.array(votes).T).tolist()

        data = np.array([t1 + t2 for t1, t2 in zip(crowddata, crowdlabel)])
        columns = [l for i, l in sorted(self.feature_dict.items())]
        self.learningset = pd.DataFrame(data, columns=columns)
        self.restset = self.dataset[~self.dataset['pid'].isin(self.learningset['pid'])]

    def Expert(self,seleteddata):
        self.learningset=seleteddata
        self.restset=self.dataset[~self.dataset['pid'].isin(self.learningset['pid'])]


    def get_feature_dict(self):
        return self.feature_dict
    def get_dataset(self):
        return self.dataset
    def get_learning_set(self):
        return self.learningset
    def get_restset(self):
        return self.restset
