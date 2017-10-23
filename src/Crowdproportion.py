import random

from collections import defaultdict
import pandas as pd
from collections import Counter

class CrowsDis:

    def __init__(self, filename):
        self.filename = filename
        self.crowdAns = defaultdict(list)
        self.crowdVote = dict()
        self.crowdvotequestion1=dict()
        self.workerforeachpaper=defaultdict(list)


    def proportion(self):
        lines=[]
        with open(self.filename) as f:

            lines = f.read().splitlines()



        for item in lines:

            info=item.split('\t')

            self.crowdAns[info[7]].append(info[9] +','+ info[10] + ',' +info[11] +','+info[12])
            self.workerforeachpaper[info[7]].append(info[1])


    def voting(self):
        for k in self.crowdAns.keys():

            Q1=0
            Q2=0
            Q3=0
            Q4=0

            for p in range(5):

                    if (self.crowdAns[k][p]).split(',')[0] == "Yes" or (self.crowdAns[k][p]).split(',')[0] =="CantTell":

                       Q1 += 1
                    if (self.crowdAns[k][p]).split(',')[1] == "Yes" or (self.crowdAns[k][p]).split(',')[1] == "CantTell":
                        Q2+=1
                    if (self.crowdAns[k][p]).split(',')[2] == "Yes" or (self.crowdAns[k][p]).split(',')[2] == "CantTell":
                        Q3+=1
                    if (self.crowdAns[k][p]).split(',')[3]!='-' and (self.crowdAns[k][p]).split(',')[3]!="NoInfo":

                            if int((self.crowdAns[k][p]).split(',')[3])>= 10:
                             Q4+=1



            if (Q1 > 2 and Q2 > 2 and Q3 > 2 and Q4 > 2):
                self.crowdVote[k]=1
            else:
                self.crowdVote[k]=0
            if Q1 > 2:
                self.crowdvotequestion1[k]=1
            else:
                self.crowdvotequestion1[k]=0



    def decisionwriting(self):

        for item in self.crowdVote.keys():
            with open('../data/voteProtonbeam.csv','w') as f:
                f.write(item+','+self.crowdVote[item]+'\n')

    def rndcrowddecision(self):

        rndC = random.sample(xrange(len(self.crowdVote)), 200)
        pos=[ y for x,y in self.crowdVote.items() if y=="1"]

        return  pos

    def workerdist(self):
        lines = []
        with open(self.filename) as f:
            lines = f.read().splitlines()
        workers = []
        for item in lines:
            workers.append(item.split('\t')[1])

        w = Counter(workers)
        for item in w :
            print str(item) +':'+str(w[item])

        return w



    def customizevoting(self, crowdnum, exclusionlist):

        print exclusionlist
        for k in self.crowdAns.keys():


            Q1 = 0
            Q2 = 0
            Q3 = 0
            Q4 = 0

            for p in range(crowdnum):
                if p in exclusionlist:
                    continue

                if (self.crowdAns[k][p]).split(',')[0] == "Yes" or (self.crowdAns[k][p]).split(',')[0] == "CantTell":
                    Q1 += 1
                if (self.crowdAns[k][p]).split(',')[1] == "Yes" or (self.crowdAns[k][p]).split(',')[1] == "CantTell":
                    Q2 += 1
                if (self.crowdAns[k][p]).split(',')[2] == "Yes" or (self.crowdAns[k][p]).split(',')[2] == "CantTell":
                    Q3 += 1
                if (self.crowdAns[k][p]).split(',')[3] != '-' and (self.crowdAns[k][p]).split(',')[3] != "NoInfo":

                    if int((self.crowdAns[k][p]).split(',')[3]) >= 10:
                        Q4 += 1

            # minvote = crowdnum - len(exclusionlist)
            # if minvote % 2 == 0: # if is even
            #     minvote = minvote/2
            # elif minvote % 2 ==1:
            #     minvote = (minvote / 2) + 1
            minvote = crowdnum - len(exclusionlist)


            if(Q1 >= minvote):
            # if (Q1 >= minvote and Q2 >= minvote and Q3 >= minvote and Q4 >= minvote):
                self.crowdVote[k] = 1
            else:
                self.crowdVote[k] = 0

        return self.crowdVote



    def getvotes(self):
        return self.crowdVote
    def getcrowdAns(self):
        return  self.crowdAns

    def getcrowdvotequestion1(self):
        return self.crowdvotequestion1




    def wallaceExpriment(self, strategy):

        for k in self.crowdAns.keys():


            Q1 = 0
            Q2 = 0
            Q3 = 0
            Q4 = 0

            for p in range(5):


                if (self.crowdAns[k][p]).split(',')[0] == "Yes" or (self.crowdAns[k][p]).split(',')[0] == "CantTell":
                    Q1 += 1
                if (self.crowdAns[k][p]).split(',')[1] == "Yes" or (self.crowdAns[k][p]).split(',')[1] == "CantTell":
                    Q2 += 1
                if (self.crowdAns[k][p]).split(',')[2] == "Yes" or (self.crowdAns[k][p]).split(',')[2] == "CantTell":
                    Q3 += 1
                if (self.crowdAns[k][p]).split(',')[3] != '-' and (self.crowdAns[k][p]).split(',')[3] != "NoInfo":

                    if int((self.crowdAns[k][p]).split(',')[3]) >= 10:
                        Q4 += 1





            if (Q1 >= strategy and Q2 >= strategy and Q3 >= strategy and Q4 >= strategy):
                self.crowdVote[k] = 1
            else:
                self.crowdVote[k] = 0

        return self.crowdVote


    def getvotesofeachworker(self):
        workers = defaultdict(dict)
        lines = []
        with open(self.filename) as f:
            lines = f.read().splitlines()
        for line in lines:
            items = line.split('\t')
            workers[items[1]][items[7]] = 1 if items[9]=='Yes' or items[9]=='CantTell' else 0
        return workers




