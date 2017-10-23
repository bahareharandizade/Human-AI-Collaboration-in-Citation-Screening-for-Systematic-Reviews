from classifier import *
import os
import pickle
import numpy as np



resultsvm=dict()
resultforrest=dict()


def machine_run(learningset,lables,restset,run,K,M):

    p_file = '../data/run-svm-{0}-{1}-{2}.pkl'
    filename = p_file.format(K,M,run)
    if os.path.exists(filename)==False:

        i=1
        for kernel in ['linear','rbf']:
            resultsvm[kernel]=dict()
            for class_weight in ['balanced']:
                resultsvm[kernel][class_weight]=dict()
                for C in ['1','100']:
                    resultsvm[kernel][class_weight][C]=dict()
                    for gamma in ['auto','10']:
                        resultsvm[kernel][class_weight][C][gamma]=Esvc(learningset,lables,restset,kernel,class_weight,C,gamma)

                        print i
                        i = i + 1

        pickle.dump(resultsvm, open(filename, 'w'))

    p_file = '../data/run-forrest-{0}-{1}-{2}.pkl'
    filename = p_file.format(K,M,run)
    if os.path.exists(filename)==False:
        j=1
        for n_estimators in ['500','1000']:
            resultforrest[n_estimators]=dict()
            for max_features in ['auto']:
                resultforrest[n_estimators][max_features]=dict()
                for class_weight in ['balanced']:
                    resultforrest[n_estimators][max_features][class_weight]=randomforrest(learningset,lables,restset,n_estimators,max_features,class_weight)

                    print j
                    j = j + 1

        pickle.dump(resultforrest, open(filename, 'w'))


def combinescore(RunNume,K,M):
    p_file = '../data/run-{0}-{1}-{2}.pkl'
    filename = p_file.format(RunNume,K,M)
    if os._exists(filename)==False:
        p_file = '../data/run-svm-{0}-{1}-{2}.pkl'
        filename = p_file.format(K,M,RunNume)
        datasvm = pickle.load(open(filename, 'r'))

        p_file = '../data/run-forrest-{0}-{1}-{2}.pkl'
        filename = p_file.format(K,M,RunNume)
        fordata = pickle.load(open(filename, 'r'))


        result=dict()
        i=1
        for kernel_key,kernel in datasvm.iteritems():
            result[kernel_key]=dict()
            for class_weight_key,class_weight in kernel.iteritems():
                result[kernel_key][class_weight_key]=dict()
                for C_key,C in class_weight.iteritems():
                    result[kernel_key][class_weight_key][C_key]=dict()
                    for gamma_key,gamma in C.iteritems():
                        q1=gamma[0]# add all score from different parameters of SVM classifier
                        q2=gamma[1]
                        result[kernel_key][class_weight_key][C_key][gamma_key]=dict()
                        j=1

                        for n_estimators_key,n_estimators in fordata.iteritems():
                            result[kernel_key][class_weight_key][C_key][gamma_key][n_estimators_key]=dict()
                            for max_features_key,max_features in n_estimators.iteritems():
                                result[kernel_key][class_weight_key][C_key][gamma_key][n_estimators_key][max_features_key]=dict()
                                for class_weight_key,class_weight in max_features.iteritems():
                                    result[kernel_key][class_weight_key][C_key][gamma_key][n_estimators_key][
                                        max_features_key][class_weight_key]=dict()
                                    p1 = class_weight[0]
                                    p2= class_weight[1]
                                    result[kernel_key][class_weight_key][C_key][gamma_key][n_estimators_key][
                                        max_features_key][class_weight_key]['svmscore']=q1
                                    result[kernel_key][class_weight_key][C_key][gamma_key][n_estimators_key][
                                        max_features_key][class_weight_key]['svmlabel']=q2
                                    result[kernel_key][class_weight_key][C_key][gamma_key][n_estimators_key][
                                        max_features_key][class_weight_key]['forrestscore'] = p1
                                    result[kernel_key][class_weight_key][C_key][gamma_key][n_estimators_key][
                                        max_features_key][class_weight_key]['forrestlabel'] = p2


                                    j=j+1

                        i=i+1

        p_file = '../data/run-{0}-{1}-{2}.pkl'
        filename = p_file.format(RunNume,K,M)
        pickle.dump(result, open(filename, 'w'))

def getmachinesScore(RunNume,K,M):
    p_file = '../data/run-{0}-{1}-{2}.pkl'
    filename = p_file.format(RunNume,K,M)

    data = pickle.load(open(filename, 'r'))

    #example--default value
    svmscore=data['rbf']['balanced']['1']['auto']['500']['auto']['balanced']['svmscore']
    svmlabel=data['rbf']['balanced']['1']['auto']['500']['auto']['balanced']['svmlabel']
    forrestscore = data['rbf']['balanced']['1']['auto']['500']['auto']['balanced']['forrestscore']
    forrestlabel = data['rbf']['balanced']['1']['auto']['500']['auto']['balanced']['forrestlabel']


    svmlabel = svmlabel.astype(np.int)
    forrestlabel = forrestlabel.astype(np.int)

    label=svmlabel+forrestlabel
    label=[1 if i==2 else i for i in label ]
    print svmlabel
    print forrestlabel
    print label
    meanscore=(svmscore+forrestscore)/2 # average probability between two model

    indexsort=np.argsort(meanscore)#get index of sort array

    return label,meanscore#numpy ndarray


