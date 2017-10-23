import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import re
import csv
from bs4 import BeautifulSoup
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import bernoulli
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold



def compute_measures(tn, fp, fn, tp):
    #tn, fp, fn, tp

    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)
    sensitivity = tp / (tp + fn)#recall for in

    specificity = tn / (tn + fp)#recall for out
    precision   = tp / (tp + fp)#precision for in
    if ((2**2 * precision) + sensitivity > 0.0) :
        f2measure = (1+2**2) * (precision * sensitivity) / ((2**2 * precision) + sensitivity)
    else:
        f2measure=-0.1
    return sensitivity, specificity, precision, f2measure

def plot_confusion_matrix(y_test, result,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.binary):
    class_names = ["Out", "In"]

    cm = confusion_matrix(y_test, result)

    np.set_printoptions(precision=2)

    res = plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=len(y_test))
    plt.title(title)
    plt.colorbar(res)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print cm
    cmflat=cm.flatten()
    precsion1, recall1, fscore1, support1 = precision_recall_fscore_support(y_test, result, labels=[-1,1])
    print precsion1
    print recall1
    print cmflat
    sensitivity, specificity, precision, f2measure = compute_measures(*cmflat)
    print "sensitivity1:%s" % recall1[1]
    print "specificity1: %s" % recall1[0]
    print "precision1: %s" % precsion1[1]

    print "\n----"

    print "sensitivity: %s" % sensitivity
    print "specificity: %s" % specificity
    print "precision: %s" % precision
    print "f2measure: %s" % f2measure

    thresh = 0.5

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 3),
                 horizontalalignment="center",
                 color="blue")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_path = '../plots/{0}.png'
    name = fig_path.format(title)
    plt.savefig(name)
    plt.clf()

    return sensitivity,specificity,precision,f2measure

def preparingtext(dataset):
    abs=dataset[[2]].values

    cleansent = []
    for sent in dataset:
        sent = " ".join(sent)
        sent = BeautifulSoup(sent, 'html.parser').get_text()
        sent = re.sub("[^a-zA-Z]", " ", sent)
        tokens = sent.lower().split()
        cleansent.append(" ".join(tokens))



    text1 = np.array([t1 + t2 for t1, t2 in zip(dataset[[1]].values.tolist(), dataset[[2]].values.tolist())])





def _load_data():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(current_dir + "/../")
    path=os.path.join(os.path.abspath(current_dir + "/../"), 'data', 'proton-beam-merged.csv')
    print path
    texts, labels, pmids = [], [], []
    csv_reader = csv.reader(open(path,'rb'))
    csv_reader.next() # skip headers
    for r in csv_reader:
        pmid, label, text = r
        texts.append(text)
        labels.append(int(label))
        pmids.append(pmid)
    return texts, labels, pmids

def classify():

    result=defaultdict(list)
    start=np.zeros(4).tolist()

    # texts=preparingtext(dataset)
    texts, labels, pmids = _load_data()
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000)

    X = vectorizer.fit_transform(texts)
    y = np.array(labels)


    kf = KFold(n_splits=10, random_state=10)
    for train, test in kf.split(X):
        
        test_size=X[test].shape[0]
    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = test_size, random_state = 42)

    # print X_train.shape[0]
    # print X_test.shape[0]
        print 'random'

        y_pred = bernoulli.rvs(.5, size=X[test].shape[0])
        for i in range(0,len(y_pred)):
            if y_pred[i] == 0:
                y_pred[i] = -1

        sensitivity, specificity, precision, f2measure = plot_confusion_matrix(y[test], y_pred, normalize=True,
                          title='PureRandom,testsize=' + str(test_size))

        if len(result['rand']) == 0:
                 result['rand']=  start


        result['rand'] = np.add(result['rand'],[sensitivity,specificity,precision,f2measure])

        print 'better than random'

        OutDist=[x for x in y[train] if x==-1]
        InDist=[x for x in y[train] if x==1]


        lenwhole= len(OutDist) + len(InDist)
        rndist=len(InDist)/float(lenwhole)




        y_pred = bernoulli.rvs(rndist, size=X[test].shape[0])
        print y_pred
        for i in range(0,len(y_pred)):
            if y_pred[i] == 0:
                y_pred[i] = -1

        sensitivity, specificity, precision, f2measure = plot_confusion_matrix(y[test], y_pred, normalize=True,
                          title='betterThanRandom,testsize=' + str(test_size))

        if len(result['brand']) == 0:
                 result['brand']=  start

        result['brand'] = np.add(result['brand'],[sensitivity, specificity, precision, f2measure])
        print 'NaiveBase'

        parameters_NaiveBase = {
                            'alpha': [1.0,0.1,0.01, 0.001],
                            'fit_prior':(True,False)

                                }

        NaiveBase_clf = MultinomialNB()
        clf = GridSearchCV(NaiveBase_clf, parameters_NaiveBase, n_jobs=-1)
        clf = clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        sensitivity, specificity, precision, f2measure = plot_confusion_matrix(y[test], y_pred,  normalize=True,
                          title='NaiveBase,testsize='+str(test_size))

        if len(result['NB']) == 0:
                 result['NB']=  start

        result['NB'] = np.add(result['NB'],[sensitivity, specificity, precision, f2measure])

        print 'KNN'

        KNN_clf = KNeighborsClassifier()
        parameters_knn = {
                    'n_neighbors': [3,5,7,10],
                    'weights': ['uniform','distance'],
                    'algorithm': ['auto'],
                    'leaf_size':[10,30,50]

                      }
        clf = GridSearchCV(KNN_clf, parameters_knn, n_jobs=-1)
        clf = clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        sensitivity, specificity, precision, f2measure = plot_confusion_matrix(y[test], y_pred, normalize=True,
                          title='KNN,testsize=' + str(test_size))

        if len(result['KNN']) == 0:
                 result['KNN']=  start

        result['KNN'] =np.add(result['KNN'],[sensitivity, specificity, precision, f2measure])

        print 'SVM'

        parameters_SVM ={
                        'kernel': ['linear','rbf'],
                        'probability':[True],
                        'C':[10,100,1000],
                        'class_weight':({1:10},'balanced'),
                        'gamma':['auto',0.01,0.1,0.001]

                        }
        SVM_clf = SVC()
        clf = GridSearchCV(SVM_clf, parameters_SVM, n_jobs=-1)
        clf = clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        sensitivity, specificity, precision, f2measure = plot_confusion_matrix(y[test], y_pred, normalize=True,
                          title='SVM,testsize=' + str(test_size))

        if len(result['SVM']) == 0:
            result['SVM'] = start
        result['SVM'] = np.add(result['SVM'],[sensitivity, specificity, precision, f2measure])

        print 'RandomForest'

        parameters_RF = {
                    'n_estimators': [10,15,20,30],
                    'class_weight' :({1:10},'balanced'),
                    'max_features': ['auto', 'log2', None],

                    }


        RF_clf = RandomForestClassifier()
        clf = GridSearchCV(RF_clf, parameters_RF, n_jobs=-1)
        clf = clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        sensitivity, specificity, precision, f2measure = plot_confusion_matrix(y[test], y_pred, normalize=True,
                          title='RF,testsize=' + str(test_size))
        if len(result['RF']) == 0:
            result['RF'] = start
        result['RF'] = np.add(result['RF'],[sensitivity, specificity, precision, f2measure])

    return result

def drawresult(result):
    print result
    x = range(1,len(result)+1,1)
    metrics={'sensitivity':0, 'specificity':1, 'precision':2, 'loss':3}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for ax,m in zip(axes.ravel(),metrics.keys()):
        print m
        y=[]
        for model in result:
            y.append(result[model][metrics[m]]/10.)
        print y




        ax.plot(x, y,
                'o',
                label=m,

                )
        ax.set_xlabel('classifiers')
        ax.set_ylabel(m)
        ax.margins(0.1, 0.1)

    plt.setp(axes, xticks=x, xticklabels=result.keys(),yticks=np.arange(-0.1,1.1,0.1))





  # writes strings with 45 degree angle

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.8)
    # axes.flatten()[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize='small')
    fig.tight_layout()
    plt.show()




#
# result = classify()
# drawresult(result)

classes=["Out", "In"]
np.set_printoptions(precision=2)
cmap=plt.cm.binary

cm= np.array([[ 4287,219],[23,220]])
res=plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=476)
plt.title('base-SGD-Wallac')
plt.colorbar(res)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
normalize=True
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')

print(cm)

thresh = 0.5
print thresh
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, round(cm[i, j],3),
             horizontalalignment="center",
             color="blue")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()





