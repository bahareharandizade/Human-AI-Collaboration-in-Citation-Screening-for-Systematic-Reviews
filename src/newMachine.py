import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import numpy as np
import random
import csv
from Crowd import *
from scoring import compute_measures
from itertools import product
import  random


def batches(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def _load_data(path):
    texts, labels, pmids = [], [], []
    csv_reader = csv.reader(open(path, 'rb'))
    csv_reader.next() # skip headers
    for r in csv_reader:
        pmid, label, text = r
        texts.append(text)
        labels.append(int(label))
        pmids.append(pmid)
    return texts, labels, pmids


def machineRun(balancing):
    texts, labels, pmids = _load_data('../data/proton-beam-merged.csv')

    labels = []
    getcrowdvotequestion1 = Cmain()  # change the label with first question label!
    for item in pmids:
        labels.append(getcrowdvotequestion1[item])
    if (balancing > 0):
        Outscope = [i for i, j in list(enumerate(labels)) if j == 0]# get index
        Inscope = [i for i, j in list(enumerate(labels)) if j == 1]# get index
        sample = len(Inscope) * balancing
        candid = random.sample(Outscope,sample)# random sample from out
        texts = [j for i, j in list(enumerate(texts)) if i in Inscope] + [j for i, j in list(enumerate(texts)) if i in candid]
        labels = [j for i, j in list(enumerate(labels)) if i in Inscope] + [j for i, j in list(enumerate(labels)) if i in candid]
        pmids = [j for i, j in list(enumerate(pmids)) if i in Inscope] + [j for i, j in list(enumerate(pmids)) if i in candid]

    print len(texts)
    print len(labels)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000,norm='l2')
    X = vectorizer.fit_transform(texts)

    X = X.toarray()
    y = np.array(labels)



    xxx = len([x for x in y if x == 0])
    print xxx
    print len(y) - xxx
    print xxx / float(len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42,stratify=y)


    xxx =  len([x for x in y_train if x==0])
    yyy = len([x for x in y_train if x == 1])
    print xxx
    print yyy

    X1 = len([x for x in y_test if x==0])
    Y1 = len([x for x in y_test if x == 1])
    print X1
    print Y1

    print xxx/float(xxx+yyy)
    print X1/float(X1+Y1)

    result=[]

    # Machine 1 DummyClassifier
    print 'DummyClassifier_stratified'
    Random_classifier = DummyClassifier(strategy='stratified',random_state=42).fit(X_train,y_train)
    y_pred = Random_classifier.predict(X_test)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    accuracy_train = Random_classifier.score(X_train,y_train)
    accuracy_test = Random_classifier.score(X_test, y_test)
    f1score = metrics.f1_score(y_test,y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test,y_pred)
    result.append(['DumClassifierStratified', accuracy_train, accuracy_test, f1score, roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)

    # Machine 1 DummyClassifier
    print 'DummyClassifier_stratified'
    Random_classifier = DummyClassifier(strategy='most_frequent', random_state=42).fit(X_train, y_train)
    y_pred = Random_classifier.predict(X_test)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    accuracy_train = Random_classifier.score(X_train, y_train)
    accuracy_test = Random_classifier.score(X_test, y_test)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['DumClassifierMostfrequent', accuracy_train, accuracy_test, f1score, roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)

    #Machine 1 NaiveBase
    print 'Machine 1 MultinomialNaiveBase'
    gs_NaiveBase_clf = MultinomialNB().fit(X_train, y_train)
    y_pred = gs_NaiveBase_clf.predict(X_test)
    accuracy_train = gs_NaiveBase_clf.score(X_train,y_train)
    accuracy_test = gs_NaiveBase_clf.score(X_test, y_test)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    f1score = metrics.f1_score(y_test,y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test,y_pred)
    result.append(['MultinomialNB',accuracy_train,accuracy_test,f1score,roc, precision, recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)

    #Machine 1 BeNaiveBase
    print 'Machine 1 BernoulliNB'
    gs_NaiveBase_clf = BernoulliNB().fit(X_train, y_train)
    y_pred = gs_NaiveBase_clf.predict(X_test)
    accuracy_train = gs_NaiveBase_clf.score(X_train,y_train)
    accuracy_test = gs_NaiveBase_clf.score(X_test, y_test)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    f1score = metrics.f1_score(y_test,y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test,y_pred)
    result.append(['BernoulliNB',accuracy_train,accuracy_test,f1score,roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)


    #Machine 2 SGD Norm2
    print 'Machine 2 SGD'
    params_d = {"alpha": 10.0 ** -np.arange(1, 7)}
    sgd = SGDClassifier(class_weight={1:1}, random_state=42,penalty='l2')
    clfsgd = GridSearchCV(sgd, params_d, scoring='roc_auc',cv=3)
    print clfsgd
    clfsgd = clfsgd.fit(X_train, y_train)
    y_pred = clfsgd.predict(X_test)
    accuracy_train = clfsgd.score(X_train,y_train)
    accuracy_test = clfsgd.score(X_test, y_test)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    f1score = metrics.f1_score(y_test,y_pred,average='macro')
    roc = metrics.roc_auc_score(y_test,y_pred)
    result.append(['SGDl2{1:1}',accuracy_train,accuracy_test,f1score,roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)

    # Machine 2 SGD Norm1
    print 'Machine 3 SGD'
    params_d = {"alpha": 10.0 ** -np.arange(1, 7)}
    sgd = SGDClassifier(class_weight={1:1},random_state=42,penalty='l1')
    clfsgd = GridSearchCV(sgd, params_d, scoring='roc_auc',cv=3)
    clfsgd = clfsgd.fit(X_train, y_train)
    y_pred = clfsgd.predict(X_test)
    accuracy_train = clfsgd.score(X_train,y_train)
    accuracy_test = clfsgd.score(X_test, y_test)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    f1score = metrics.f1_score(y_test,y_pred,average='macro')
    roc = metrics.roc_auc_score(y_test,y_pred)
    result.append(['SGDl1{1:1}',accuracy_train,accuracy_test,f1score,roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)


    # print 'Machine 4 Mini_Batch GD'
    # sgd = SGDClassifier(class_weight={1:10},random_state=42, penalty='elasticnet',alpha=clfsgd.best_estimator_.alpha)
    # shuffledRange = range(len(X_train))
    # n_iter = 100 #epoche
    # for n in range(n_iter):
    #     random.shuffle(shuffledRange)
    #     shuffledX = [X_train[i] for i in shuffledRange]
    #     shuffledY = [y_train[i] for i in shuffledRange]
    #     for batch in batches(range(len(shuffledX)), 2000):
    #         sgd.partial_fit(shuffledX[batch[0]:batch[-1] + 1], shuffledY[batch[0]:batch[-1] + 1],classes=np.unique(y_train))
    #
    # y_pred = sgd.predict(X_test)
    # accuracy_train = sgd.score(X_train, y_train)
    # accuracy_test = sgd.score(X_test, y_test)
    # f1score = metrics.f1_score(y_test, y_pred)
    # roc = metrics.roc_auc_score(y_test, y_pred)
    # result.append(['Minibatch_GD', accuracy_train, accuracy_test, f1score, roc])


    #Machine 3 RandomForrest

    print 'Machine 4 RandomForrest'
    RF_clf =  RandomForestClassifier(class_weight={1:5},random_state=42)
    parameters_RF = {
                        'n_estimators':[300],# 300 is enough
                        'max_depth': [20]# this is good fit
                    }

    gs_RF_clf = GridSearchCV(RF_clf, parameters_RF, n_jobs=-1,scoring='roc_auc',cv=3)
    gs_RF_clf = gs_RF_clf.fit(X_train, y_train)
    print 'RF fitted!'
    print gs_RF_clf.best_estimator_
    y_pred = gs_RF_clf.predict(X_test)
    accuracy_train = gs_RF_clf.score(X_train,y_train)
    accuracy_test = gs_RF_clf.score(X_test, y_test)
    f1score = metrics.f1_score(y_test,y_pred,average='macro')
    roc = metrics.roc_auc_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    result.append(['RF{1:5}',accuracy_train,accuracy_test,f1score,roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)
    #
    #Machine 4 KNN
    print 'Machine 5 KNN'
    knn_clf =  KNeighborsClassifier(weights='uniform')


    parameters_knn = {
                    'n_neighbors': [2,3,4]
                      }
    gs_knn_clf = GridSearchCV(knn_clf, parameters_knn,scoring='roc_auc', n_jobs=-1,cv=3)
    gs_knn_clf = gs_knn_clf.fit(X_train, y_train)
    y_pred = gs_knn_clf.predict(X_test)
    accuracy_train = gs_knn_clf.score(X_train,y_train)
    accuracy_test = gs_knn_clf.score(X_test, y_test)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    f1score = metrics.f1_score(y_test,y_pred)
    roc = metrics.roc_auc_score(y_test,y_pred)
    result.append(['KNN',accuracy_train,accuracy_test,f1score,roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)
    # #
    # # Machine 4 GB
    print 'Machine 6 GB'
    GB_clf = GradientBoostingClassifier(random_state=42)

    parameters_GB = {
        'n_estimators':[100],
        'learning_rate': [0.1],



    }

    gb_clf = GridSearchCV(GB_clf, parameters_GB,scoring='roc_auc',n_jobs=-1,cv=3)
    gb_clf = gb_clf.fit(X_train, y_train)
    print 'GB fitted!'
    print gb_clf.best_params_
    y_pred = gb_clf.predict(X_test)
    accuracy_train = gb_clf.score(X_train, y_train)
    accuracy_test = gb_clf.score(X_test, y_test)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    f1score = metrics.f1_score(y_test, y_pred,average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['GB',accuracy_train, accuracy_test, f1score, roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)

    # Machine 4 GB
    # print 'Machine 7 Neural Network'
    # ML_clf = MLPClassifier(activation='tanh',solver='lbfgs',random_state=42,hidden_layer_sizes=[100,100,100], alpha=0.01)
    #
    # # parameters_MLPC = {
    # #     'hidden_layer_sizes':[(10,)],
    # #     'alpha': [0.001],
    # #
    # #
    # #
    # # }
    #
    # # NN_clf = GridSearchCV(ML_clf, parameters_MLPC,scoring='roc_auc',n_jobs=-1)
    # NN_clf = ML_clf.fit(X_train, y_train)
    # print 'NN fitted!'
    #
    # y_pred = NN_clf.predict(X_test)
    # accuracy_train = NN_clf.score(X_train, y_train)
    # accuracy_test = NN_clf.score(X_test, y_test)
    # f1score = metrics.f1_score(y_test, y_pred,average='macro')
    # roc = metrics.roc_auc_score(y_test, y_pred)
    # result.append(['Neural Network',accuracy_train, accuracy_test, f1score, roc])
    # # print NN_clf.best_params_
    # print 'accuracy_train:' + str(accuracy_train)
    # print 'accuracy_test:' + str(accuracy_test)
    # print 'f1score:' + str(f1score)
    # print 'roc_auc_score:' + str(roc)

    print 'Machine 6 baggingWithSVC'
    n_estimators= 10
    SVC_clf = BaggingClassifier(base_estimator=SVC(kernel='linear',class_weight={1:10}),n_estimators=n_estimators,max_samples=1.0 /n_estimators,random_state=42, max_features=0.3)


    SVC_clf = SVC_clf.fit(X_train, y_train)
    print 'baggingWithSVC fitted!'

    y_pred = SVC_clf.predict(X_test)
    accuracy_train = SVC_clf.score(X_train, y_train)
    accuracy_test = SVC_clf.score(X_test, y_test)
    f1score = metrics.f1_score(y_test, y_pred,average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    result.append(['SVCBagging{1:10}',accuracy_train, accuracy_test, f1score, roc,precision,recall])
    print 'accuracy_train:' + str(accuracy_train)
    print 'accuracy_test:' + str(accuracy_test)
    print 'f1score:' + str(f1score)
    print 'roc_auc_score:' + str(roc)
    print 'recall:'+str(recall)
    print 'precision:'+str(precision)


    return result


def correlation(clf_names,y_pred,y_true):
    StateCode = {0:'tn',1:'fp',2:'tp',3:'fn'}
    total = len(y_pred)
    results = {}
    machineCombination = []
    for i in xrange(2, len(clf_names)+1):
        machineCombination.append(list(itertools.combinations(clf_names, i)))# combination of classifier

    for mlist in machineCombination:
        for com in mlist:

            key = ''
            partialans = []
            A1= []
            A0=[]
            A1.append(list(product([2, 3], repeat=len(com))))
            A0.append(list(product([0, 1], repeat=len(com))))
            Classifiers = []
            for item in com:
                key+=item+'-'
                Classifiers.append(clf_names.index(item))
            for acts in A0[0]:
                X = 1
                Y = 0
                for act,C in zip(acts,Classifiers):
                    X*=(StateCode[act][C]/total)
                    Y+=StateCode[act][C]
                X*=total
                partialans.append((((Y-X)**2)/X))

            for acts in A1[0]:
                X = 1
                Y = 0
                for act,C in zip(acts,Classifiers):
                    X*=(StateCode[act][C]/total)
                    Y+=StateCode[act][C]
                X*=total
                partialans.append((((Y-X)**2)/X))


            results[key] = np.sum(partialans)





    print results





# make some plots

results = machineRun(2)
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(7)]
print results

clf_names, accuracy_train, accuracy_test, f1score, roc,precision,recall = results

# correlation(clf_names,y_pred,y_test)




plt.figure(figsize=(8, 8))
plt.title("Score:Balanced{1:3},OutScope is two times as InScope")
plt.barh(indices, accuracy_train, .06, label="accuracy_train", color='navy',align='center')
plt.barh(indices + .1, accuracy_test, .06, label="accuracy_test",
         color='c',align='center')
plt.barh(indices + .2, f1score, .06, label="f1score", color='darkorange',align='center')
plt.barh(indices + .3, roc, .06, label="ROC_AUC", color='red',align='center')
plt.barh(indices + .4, precision, .06, label="precision", color='green',align='center')
plt.barh(indices + .5, recall, .06, label="recall", color='y',align='center')
plt.yticks(())

plt.legend(loc='best',fontsize='small')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)
plt.xticks(np.arange(0.0, 1.06, 0.05))
plt.show()



# avval bia begoo ehtemal TP,FP,TN,FN to har kodom cheghadre baad bia 2 be 2 begoo majmoo TP dar har do dar halati ke hast. baad halati mostaghel bashe yani TP(c1)*Tp(c2)*tedad