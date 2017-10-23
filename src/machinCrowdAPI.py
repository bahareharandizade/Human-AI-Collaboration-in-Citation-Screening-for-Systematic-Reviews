import numpy as np
from Crowd import *
from scoring import compute_measures
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from APImachine import *
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from collections import defaultdict


def workerByIdMachine():
    workerlist = getworkerdistribution()
    workerlist = dict((i,v) for i,v in workerlist.iteritems() if v>=30)# filter workers with more that 30 papers
    goldvotes = Cmain()# get gold based on first question!
    crowdand = getvoteforeachpaper()
    result=defaultdict(list)

    X, pmids, X_train, y_train = maketraingset()
    clfsgd, gs_SVM_clf, gs_knn_clf, gs_RF_clf = makemachines(X_train,y_train)
    print 'tamam'
    for worker in workerlist:
        print worker
        w = crowdand[worker]#the paper list of this worker
        g = dict((i,v) for i,v in goldvotes.iteritems() if i in w.keys())# the gold data for the paper this worker vote on
        worker_ans = []
        gold = []
        machinesdg = []
        machinesvm = []
        machinerf = []
        machineknn = []

        for paper in g:
            worker_ans.append(w[paper])
            gold.append(g[paper])
            paperindex = pmids.index(paper)
            textpaper = X[paperindex]

            machinesdg.append(clfsgd.predict(textpaper)[0])
            machinesvm.append(gs_SVM_clf.predict(textpaper)[0])
            machinerf.append(gs_RF_clf.predict(textpaper)[0])
            machineknn.append(gs_knn_clf.predict(textpaper)[0])
            # machinenb.append(gs_NaiveBase_clf.predict(textpaper)[0])

        print worker
        print gold
        print worker_ans
        print machinesdg
        print machinesvm
        print machinerf
        print machineknn


        cm = confusion_matrix(gold, worker_ans).flatten()
        result[worker].append(compute_measures(*cm))#measure for this worker

        cm = confusion_matrix(gold, machinesdg).flatten()
        result[worker].append(compute_measures(*cm))

        cm = confusion_matrix(gold, machinesvm).flatten()
        result[worker].append(compute_measures(*cm))

        cm = confusion_matrix(gold, machinerf).flatten()
        result[worker].append(compute_measures(*cm))

        cm = confusion_matrix(gold, machineknn).flatten()
        result[worker].append(compute_measures(*cm))






    return result



def omitworkers():
    workerlist = getworkerdistribution()
    workerlist = dict((i, v) for i, v in workerlist.iteritems() if v >= 30)
    workers_paper=getworkereachpaper()#for each paper we have list of its workers
    crowdand = getvoteforeachpaper()#each worker has a list of paper he vote on
    goldvotes = Cmain()  # get gold based on first question!
    result = {}
    clf, X, pmids= maketraingset()
    for worker in workerlist:
        papers = crowdand[worker]#the paper list of this worker
        paperupate=[]
        for paper in papers.keys():
            paperindex = pmids.index(paper)
            textpaper = X[paperindex]
            vote = clf.predict(textpaper)
            workers=workers_paper[paper]#the worker list of this paper
            index = workers.index(worker)#find index if this worker
            paperupate.append([paper,index,vote[0]])
        newvote=updatevote(paperupate)#update the votes of these papers!
        g = []
        w = []
        for item in goldvotes:
            g.append(goldvotes[item])
            w.append(newvote[item])
        cm = confusion_matrix(g, w).flatten()
        result[worker] = compute_measures(*cm)


    return result


def drawresult(result):
    # key = ['whole_W','4W', '3W','2W','1W']
    #key = ['wo5', 'wo4', 'wo3', 'wo2', 'wo1','machine']
    x = range(1,len(result)+1,1)
    color = {0:'ro',1:'bx',2:'go',3:'r*',4:'gx'}
    state={0:'crowd',1:'SGD',2:'svm',3:'RF',4:'knn'}
    # metrics={'sensitivity(recall_Inscope)':0, 'specificity(recall_Outscope)':1, 'precision(precision_Inscope)':2, 'loss(fp+(R*fn))':3}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))


    ax0, ax1, ax2, ax3 = axes.flatten()

    result = OrderedDict(sorted(result.items(), key=lambda x: x[1][0][0]))
    for item in range(0,5):
        y = []
        for model in result:

            y.append(result[model][item][0])
        print x
        print y
        ax0.plot(x,y,color[item],label=state[item])

    # ax0.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    # ax0.legend(loc='upper left',fontsize='small')
    # ax0.set_xlabel('workers_Combination')
    ax0.set_ylabel('sensitivity(recall_Inscope)')
    # ax0.set_ylim((0.60, 0.95))
    # ax0.set_yticks(np.arange(0.60,0.95,0.05))
    ax0.margins(0.1, 0.1)
    ax0.label = 'sensitivity(recall_Inscope)'
    ax0.set_xticks(x)
    ax0.set_xticklabels([i[0:3] for i in result.keys()])

    result = OrderedDict(sorted(result.items(), key=lambda x: x[1][0][1]))

    for item in range(0,5):
        y = []
        for model in result:

            y.append(result[model][item][1])

        ax1.plot(x,y,color[item],label=state[item])

    # ax1.legend(loc='upper left',fontsize='small')
    # ax1.set_xlabel('workers_Combination')
    ax1.set_ylabel('specificity(recall_Outscope)')
    # ax1.set_ylim((0.90, 1.0))
    # ax1.set_yticks(np.arange(0.90, 1.0, 0.01))
    ax1.margins(0.1, 0.1)
    ax1.label = 'specificity(recall_Outscope)'
    ax1.set_xticks(x)
    ax1.set_xticklabels([i[0:3] for i in result.keys()])

    result = OrderedDict(sorted(result.items(), key=lambda x: x[1][0][2]))

    for item in range(0,5):
        y = []
        for model in result:

            y.append(result[model][item][2])

        ax2.plot(x,y,color[item],label=state[item])

    # ax2.legend(loc='upper left',fontsize='small')
    # ax2.set_xlabel('workers_Combination')
    ax2.set_ylabel('precision(precision_Inscope)')
    # ax2.set_ylim(0.3,0.60)
    # ax2.set_yticks(np.arange(0.30, 0.60, 0.02))
    ax2.margins(0.1, 0.1)
    ax2.label = 'precision(precision_Inscope)'
    ax2.set_xticks(x)
    ax2.set_xticklabels([i[0:3] for i in result.keys()])

    result = OrderedDict(sorted(result.items(), key=lambda x: x[1][0][3]))

    for item in range(0,5):
        y = []
        for model in result:

            y.append(result[model][item][3])

        ax3.plot(x,y,color[item],label=state[item])

    # ax3.legend(loc='upper left',fontsize='small')
    # ax3.set_xlabel('workers_Combination')
    ax3.set_ylabel('loss(fp+(R*fn)/n)')
    # ax3.set_ylim(500, 1150)
    # ax3.set_yticks(np.arange(500, 1150, 50))
    ax3.margins(0.1, 0.1)
    ax3.label = 'loss(fp+(R*fn)/n)'
    ax3.set_xticks(x)
    ax3.set_xticklabels([i[0:3] for i in result.keys()])


    #
    # for ax,m in zip(axes.ravel(),metrics.keys()):
    #
    #     y=[]
    #     for model in result:
    #         y.append(result[model][metrics[m]])
    #
    #     ax.plot(x, y,
    #             'o',
    #             label=m,
    #
    #             )
    #     ax.set_xlabel('workers_Combination')
    #     ax.set_ylabel(m)
    #     print y
    #     ax.set_yticks(ax4.get_ylim(np.arange(-0.1,1.1,0.1))
    #     ax.margins(0.1, 0.1)
    #
    #

    # plt.setp(axes, xticks=x, xticklabels=[i[0:3] for i in result.keys()])
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.8)
    axes.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize='small')
    fig.tight_layout()
    plt.show()



def killingWorkers():

    crowdlist = [0,1,2,3,4]
    #uncomment this if your gold data is expert idea
    # y_real,pid = getexpertvotes_label()#get y_real and order of papers!



    pid =[]
    y_real = []
    crowdvotequestion1=Cmain()#this is gold data here!
    for item in crowdvotequestion1:
        pid.append(item)
        y_real.append(crowdvotequestion1[item])
    print 'len in scope here!::'+str(len([x for x in y_real if x==1]))
    votes = getcustomizevoting(crowdlist,pid)
    print 'y_real'
    print y_real

    result={}
    #uncomment this if you want votes of all crowd!
    # allcrowdvote=Cmain()# get all crowd votes!
    # label=[]
    # for item in pid:
    #     label.append(allcrowdvote[item])
    # cm = confusion_matrix(y_real, label).flatten()
    # print 'all'
    # result[0] = compute_measures(*cm)
    # print result

    for combination in votes:
        cm = np.zeros(2 ** 2)
        print len(votes[combination])
        for index in range(len(votes[combination])):
            print votes[combination][index]
            #uncomment this if you want mean combination in each group
            # cm += confusion_matrix(y_real, votes[combination][index]).flatten()
            cm = confusion_matrix(y_real, votes[combination][index]).flatten()
            result[index] = compute_measures(*cm)
        #uncomment this if you want mean combination in each group
        #result[combination+1] = compute_measures(*cm / float(len(votes[combination]))) # average
    return result

#result = killingWorkers()


result = workerByIdMachine()
# result['machine']=oneSGDmachine()


# for item in result:
#
#     # print workersID[item]
#     print item
#     print 'sensitivity(recall_Inscope):'+str(result[item][0])
#     print 'specificity(recall_outScope):'+str(result[item][1])
#     print 'precision(precison_InScope):'+str(result[item][2])
#     print 'loss(FP+10*FN):' + str(result[item][3])
print result
drawresult(result)

# result,resultm = omitworkers()
# drawresult(result,resultm)









