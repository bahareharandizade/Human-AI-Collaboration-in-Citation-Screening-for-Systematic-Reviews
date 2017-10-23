from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from matplotlib import pyplot as plt





def analyzeresult():

    for state in ['random']:

        for K in [10]:
            p_file = '../data/{0}-K:{1}.pkl'
            filename = p_file.format(state, K)
            data = pickle.load(open(filename, 'r'))
            result= dict()
            for K_key,K_item in data.iteritems():

                result[K_key]=dict()
                for M_key,M_item in K_item.iteritems():

                    result[K_key][M_key]=dict()

                    resultlabel = defaultdict(dict)
                    resultscore=defaultdict(dict)

                    keys=sorted(M_item.keys())
                    for key in keys:# iterate over run



                        r_label=M_item[key]['real']['reallabel']
                        print key
                        print r_label
                        print len(r_label)
                        print len([x for x in r_label if x==1])
                        for keyl,slable in M_item[key]['labelAll'].iteritems():
                            print keyl
                            print slable
                            print len([x for x in slable if x==1])
                            print len([(x,y) for (x,y) in zip(r_label,slable) if x==y and x==0])
                            print len([(x, y) for (x, y) in zip(r_label, slable) if x == y and x == 1])
                            precsion,recall,fscore,support=precision_recall_fscore_support(r_label,slable,labels=[0,1])
                            p,r,s,x=precision_recall_fscore_support(r_label,slable,average='macro')# Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                            pp,rr,ss,x=precision_recall_fscore_support(r_label,slable,average='weighted')#Calculate metrics for each label, and find their average
                            print 'precision'
                            print precsion[0]
                            print precsion[1]
                            print 'recall'
                            print recall[0]
                            print recall[1]

                            if 'precsion' in resultlabel[keyl]:
                                resultlabel[keyl]['precsion'].append([precsion[0],precsion[1],p,pp])

                            else:
                                resultlabel[keyl]['precsion']=list()
                                resultlabel[keyl]['precsion'].append([precsion[0],precsion[1],p,pp])



                            if 'recall' in resultlabel[keyl]:
                                resultlabel[keyl]['recall'].append([recall[0],recall[1],r,rr])
                            else:
                                resultlabel[keyl]['recall'] = list()
                                resultlabel[keyl]['recall'].append([recall[0],recall[1],r,rr])

                            if 'fscore' in resultlabel[keyl]:
                                resultlabel[keyl]['fscore'].append([fscore[0],fscore[1],s,ss])
                            else:
                                resultlabel[keyl]['fscore'] = list()
                                resultlabel[keyl]['fscore'].append([fscore[0],fscore[1],s,ss])


                        for keyl,slable in M_item[key]['scoreAll'].iteritems():



                            precsion,recall,fscore,support=precision_recall_fscore_support(r_label,slable,labels=[0,1])
                            p,r,s,x=precision_recall_fscore_support(r_label,slable,average='macro')# Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                            pp,rr,ss,xx=precision_recall_fscore_support(r_label,slable,average='weighted')
                        #Calculate metrics for each label, and find their average




                            if 'precsion' in resultscore[keyl]:
                                resultscore[keyl]['precsion'].append([precsion[0],precsion[1],p,pp])
                            else:
                                resultscore[keyl]['precsion']=list()
                                resultscore[keyl]['precsion'].append([precsion[0],precsion[1],p,pp])


                            if 'recall' in resultscore[keyl]:
                                resultscore[keyl]['recall'].append([recall[0],recall[1],r,rr])
                            else:
                                resultscore[keyl]['recall'] = list()
                                resultscore[keyl]['recall'].append([recall[0],recall[1],r,rr])

                            if 'fscore' in resultscore[keyl]:
                                resultscore[keyl]['fscore'].append([fscore[0],fscore[1],s,ss])
                            else:
                                resultscore[keyl]['fscore'] = list()
                                resultscore[keyl]['fscore'].append([fscore[0],fscore[1],s,ss])

                    print resultlabel
                    result[K_key][M_key]['label']=dict()
                    result[K_key][M_key]['label']=resultlabel
                    result[K_key][M_key]['score']=dict()
                    result[K_key][M_key]['score']=resultscore

    return result


def drawplots(anaresult):
    subtile={0:'Outscope',1:'Inscope',2:'Avg_Macro',3:'Avg_Weighted'}
    scorekeys=set()
    fig_path = '../plots/{0}-K:{1}-M:{2}-{3}-state:{4}.png'
    for j in range(0,4):


        for K_key,K_value in anaresult.iteritems():
            for M_key,M_value in K_value.iteritems():
                labelP = list()
                labelR = list()
                labelF = list()

                scoreP = list()
                scoreR = list()
                scoreF = list()

                for state_key,state in M_value.iteritems():
                    if(state_key=='label'):
                        for lstate_key,lstate in state.iteritems():#lable1,label2,...
                            print lstate_key
                            p=[]
                            for i in range(0,len(lstate['precsion'])):
                                p.append(lstate['precsion'][i][j])
                            print p
                            labelP.append(p)

                            r=[]
                            for i in range(0, len(lstate['recall'])):
                                r.append(lstate['recall'][i][j])

                            labelR.append(r)

                            f = []
                            for i in range(0, len(lstate['fscore'])):
                                f.append(lstate['fscore'][i][j])

                            labelF.append(f)


                    elif(state_key == 'score'):
                        for lstate_key, lstate in state.iteritems():  # score0.55,label0.6,...
                            print lstate_key
                            scorekeys.add(lstate_key[lstate_key.index('0'):])# to keep the order
                            p = []
                            for i in range(0, len(lstate['precsion'])):
                                p.append(lstate['precsion'][i][j])

                            scoreP.append(p)

                            r = []
                            for i in range(0, len(lstate['recall'])):
                                r.append(lstate['recall'][i][j])

                            scoreR.append(r)

                            f = []
                            for i in range(0, len(lstate['fscore'])):
                                f.append(lstate['fscore'][i][j])

                            scoreF.append(f)

                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))

                for ax,cnt,tlabel in zip(axes.ravel(),['precision','recall','fscore'],[labelP,labelR,labelF]):

                # plottling the histograms
                    X=range(1,len(tlabel[0])+1)#runtime
                    for t in range(0, len(tlabel)):
                        ax.plot(X,tlabel[t],
                              'o',
                            label='M %s' % str(t+1),
                        )



                    ax.margins(0.2)
                    ax.set_xlabel('run')
                    ax.set_ylabel(cnt)
                    ax.set_title('label #%s' % cnt)

                fig.suptitle(subtile[j], x=0.1, y=0.99)
                fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.8)
                axes.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4,fontsize = 'small')
                fig.tight_layout()

                name = fig_path.format('random', K_key, M_key,'label',j)
                plt.savefig(name)
                plt.clf()

                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))

                for ax, cnt, tscore in zip(axes.ravel(), ['precision', 'recall', 'fscore'], [scoreP, scoreR, scoreF]):


                    X = range(1, len(tscore[0])+1)  # runtime


                    for t in range(0, len(tscore)):

                        ax.plot(X, tscore[t],
                              'o',
                              label=list(scorekeys)[t]
                        )



                    ax.margins(0.2)
                    ax.set_xlabel('run')
                    ax.set_ylabel(cnt)
                    ax.set_title('score #%s' % cnt)
                fig.suptitle(subtile[j],x=0.1, y=0.99)
                fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.8)
                axes.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize = 'small')
                fig.tight_layout()
                name = fig_path.format('random', K_key, M_key, 'score', j)
                plt.savefig(name)
                plt.clf()


def analyzemachine():

    for state in ['random']:

        for K in [10]:
            p_file = '../data/{0}-K:{1}.pkl'
            filename = p_file.format(state, K)
            data = pickle.load(open(filename, 'r'))
            result = dict()
            for K_key,K_item in data.iteritems():
                result[K_key]=dict()
                for M_key,M_item in K_item.iteritems():
                    result[K_key][M_key]=dict()
                    resultlabel = defaultdict(dict)
                    keys = sorted(M_item.keys())
                    for key in keys:# iterate over run
                        r_label = M_item[key]['real']['reallabel']  # get the real label
                        r_label = [str(x) for x in r_label]#make the label string


                        for keyl, slable in M_item[key].iteritems():
                            if keyl in ['NB','SVM','RF','knn']:
                                precsion,recall,fscore,support=precision_recall_fscore_support(r_label,slable['label'],labels=[0,1])
                                p,r,s,x=precision_recall_fscore_support(r_label,slable['label'],average='macro')# Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                                pp,rr,ss,x=precision_recall_fscore_support(r_label,slable['label'],average='weighted')#Calculate metrics for each label, and find their average


                                if 'precsion' in resultlabel[keyl]:
                                    resultlabel[keyl]['precsion'].append([precsion[0],precsion[1],p,pp])

                                else:
                                    resultlabel[keyl]['precsion']=list()
                                    resultlabel[keyl]['precsion'].append([precsion[0],precsion[1],p,pp])



                                if 'recall' in resultlabel[keyl]:
                                    resultlabel[keyl]['recall'].append([recall[0],recall[1],r,rr])
                                else:
                                    resultlabel[keyl]['recall'] = list()
                                    resultlabel[keyl]['recall'].append([recall[0],recall[1],r,rr])

                                if 'fscore' in resultlabel[keyl]:
                                    resultlabel[keyl]['fscore'].append([fscore[0],fscore[1],s,ss])
                                else:
                                    resultlabel[keyl]['fscore'] = list()
                                    resultlabel[keyl]['fscore'].append([fscore[0],fscore[1],s,ss])





                    result[K_key][M_key]['label']=dict()
                    result[K_key][M_key]['label']=resultlabel


    return result


def drawplotsmachine(anaresult):
    fig_path = '../plots/machine-{0}-K:{1}-M:{2}-state:{3}.png'
    machinekey=set()
    for j in range(0, 4):


        for K_key, K_value in anaresult.iteritems():
            for M_key, M_value in K_value.iteritems():
                labelP = list()
                labelR = list()
                labelF = list()
                for state_key, state in M_value.iteritems():  # label

                    for lstate_key, lstate in state.iteritems():  # machine1,machine2,...
                        machinekey.add(lstate_key)
                        p = []
                        for i in range(0, len(lstate['precsion'])):
                            p.append(lstate['precsion'][i][j])

                        labelP.append(p)

                        r = []
                        for i in range(0, len(lstate['recall'])):
                            r.append(lstate['recall'][i][j])

                        labelR.append(r)

                        f = []
                        for i in range(0, len(lstate['fscore'])):
                            f.append(lstate['fscore'][i][j])

                        labelF.append(f)

                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))

                for ax, cnt, tlabel in zip(axes.ravel(), ['precision', 'recall', 'fscore'], [labelP, labelR, labelF]):

                # plottling the histograms
                    X = range(1, len(tlabel[0]) + 1)  # runtime
                    for t in range(0, len(tlabel)):
                        ax.plot(X, tlabel[t],
                            'o',
                            label=list(machinekey)[t],
                        )

                    ax.margins(0.2)
                    ax.set_xlabel('run')
                    ax.set_ylabel(cnt)
                    ax.set_title(cnt)

                fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.8)
                axes.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4,fontsize = 'small')
                fig.tight_layout()
                name = fig_path.format('random', K_key, M_key, j)
                plt.savefig(name)
                plt.clf()



def drawplotsversusM(anaresult):
    scorekeys=set()
    fig_path = '../plots/{0}-K:{1}-{2}:{3}-state:{4}.png'
    for j in range(0,4):


        for K_key,K_value in anaresult.iteritems():

            Mval=set()
            labelPP = list()
            labelRR = list()
            labelFF = list()

            scorePP = list()
            scoreRR = list()
            scoreFF = list()
            MM = len(K_value)

            for M_key,M_value in K_value.iteritems():
                print M_key
                Mval.add(M_key)
                labelP = list()
                labelR = list()
                labelF = list()

                scoreP = list()
                scoreR = list()
                scoreF = list()

                for state_key,state in M_value.iteritems():

                    if(state_key=='label'):
                        lenlable=len(state)

                        for lstate_key,lstate in state.iteritems():#lable1,label2,...

                            p=[]
                            for i in range(0,len(lstate['precsion'])):
                                p.append(lstate['precsion'][i][j])

                            labelP.append(p)

                            r=[]
                            for i in range(0, len(lstate['recall'])):
                                r.append(lstate['recall'][i][j])

                            labelR.append(r)

                            f = []
                            for i in range(0, len(lstate['fscore'])):
                                f.append(lstate['fscore'][i][j])

                            labelF.append(f)


                    elif(state_key == 'score'):
                        lenscore=len(state)
                        for lstate_key, lstate in state.iteritems():  # score0.55,label0.6,...

                            scorekeys.add(lstate_key[lstate_key.index('0'):])# to keep the order
                            p = []
                            for i in range(0, len(lstate['precsion'])):
                                p.append(lstate['precsion'][i][j])

                            scoreP.append(p)

                            r = []
                            for i in range(0, len(lstate['recall'])):
                                r.append(lstate['recall'][i][j])

                            scoreR.append(r)

                            f = []
                            for i in range(0, len(lstate['fscore'])):
                                f.append(lstate['fscore'][i][j])

                            scoreF.append(f)

                print labelP
                labelFF.append(labelF)
                labelPP.append(labelP)
                labelRR.append(labelR)

                scoreFF.append(scoreF)
                scorePP.append(scoreP)
                scoreRR.append(scoreR)

            labelFF=np.array(labelFF)
            labelRR=np.array(labelRR)
            labelPP=np.array(labelPP)

            print labelPP
            for t in range(0,lenlable):
                print 'label'
                print t
                # print labelPP[0:MM][t]
                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))

                for ax, cnt, tlabel in zip(axes.ravel(), ['precision', 'recall', 'fscore'], [labelPP[:,t], labelRR[:,t], labelFF[:,t]]):

                # plottling the histograms
                    X = range(1, len(tlabel[0]) + 1)  # runtime
                    for a in range(0, len(tlabel)):
                        ax.plot(X, tlabel[a],
                            'o',
                            label=list(Mval)[a],
                            )

                    ax.margins(0.2)
                    ax.set_xlabel('run')
                    ax.set_ylabel(cnt)
                    ax.set_title('label #%s' % cnt)

                fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.8)
                axes.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize='small')
                fig.tight_layout()
                name = fig_path.format('random', K_key, 'label', t,j)
                plt.savefig(name)
                plt.clf()

            scorePP=np.array(scorePP)
            scoreRR=np.array(scoreRR)
            scoreFF=np.array(scoreFF)

            for t in range(0,lenscore):
                print 'score'
                print t
                fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))

                for ax, cnt, tscore in zip(axes.ravel(), ['precision', 'recall', 'fscore'], [scorePP[:,t], scoreRR[:,t], scoreFF[:,t]]):

                    X = range(1, len(tscore[0]) + 1)  # runtime

                    for a in range(0, len(tscore)):
                        ax.plot(X, tscore[a],
                            'o',
                            label=list(Mval)[a]
                            )

                    ax.margins(0.2)
                    ax.set_xlabel('run')
                    ax.set_ylabel(cnt)
                    ax.set_title('score #%s' % cnt)

                fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.8)
                axes.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize='small')
                fig.tight_layout()
                name = fig_path.format('random', K_key, 'score', t,j)
                plt.savefig(name)
                plt.clf()


anaresult=analyzeresult()
drawplots(anaresult)
# manaresult=analyzemachine()
# drawplotsmachine(manaresult)
# drawplotsversusM(anaresult)






