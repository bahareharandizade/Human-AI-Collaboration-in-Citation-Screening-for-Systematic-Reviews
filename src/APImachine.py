from vectorizedocs import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import csv
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from Crowd import *
from scoring import *


def machineRun(learning_set,learning_lables,rest_set,tokenizer,run,K,M):

    learning_set, rest_set = vectorize.TfidfVectorizer1(learning=learning_set,resting=rest_set, tokenizer=tokenizer, min_df=3, max_features=50000)
    # learning_set=tokenizer(learning_set)
    # rest_set=tokenizer(rest_set)

    result=dict()
    #Machine 1 NaiveBase
    print 'Machine 1 NaiveBase'
    NaiveBase_clf = MultinomialNB()


    parameters_NaiveBase = {
                            'alpha': [1.0,0.1,0.01, 0.001],
                            'fit_prior':(True,False)

                            }



    gs_NaiveBase_clf = GridSearchCV(NaiveBase_clf, parameters_NaiveBase, n_jobs=-1)
    gs_NaiveBase_clf = gs_NaiveBase_clf.fit(learning_set, learning_lables)

    print 'NaiveBase fitted!'

    result['NB']=dict()
    result['NB']['best_score']=gs_NaiveBase_clf.best_score_
    result['NB']['best_parameter']=gs_NaiveBase_clf.best_params_
    result['NB']['label']=gs_NaiveBase_clf.predict(rest_set)
    result['NB']['score'] = gs_NaiveBase_clf.predict_proba(rest_set)[:,0]
    # print result['NB']['label'].tolist()
    # print result['NB']['score'].tolist()
    print result['NB']['best_score']
    print result['NB']['best_parameter']

    #Machine 2 SVM
    print 'Machine 2 SVM'
    SVM_clf = SVC()

    parameters_SVM ={
                        'kernel': ['linear','rbf','poly','sigmoid'],
                        'probability':[True],
                        'C':[10,100,1000,10000],
                        'class_weight':({1:10},'balanced'),
                        'gamma':['auto',0.01,0.1,0.001,1]

                        }



    gs_SVM_clf = GridSearchCV(SVM_clf, parameters_SVM, n_jobs=-1)
    gs_SVM_clf = gs_SVM_clf.fit(learning_set, learning_lables)

    print 'SVM fitted!'

    result['SVM']=dict()
    result['SVM']['best_score']=gs_SVM_clf.best_score_
    result['SVM']['best_parameter']=gs_SVM_clf.best_params_
    result['SVM']['label']=gs_SVM_clf.predict(rest_set)
    result['SVM']['score'] = gs_SVM_clf.predict_proba(rest_set)[:,0]
    # print result['SVM']['label'].tolist()
    # print result['SVM']['score'].tolist()
    print result['SVM']['best_score']
    print result['SVM']['best_parameter']


    #Machine 3 RandomForrest
    #
    print 'Machine 3 RandomForrest'
    RF_clf =  RandomForestClassifier()



    parameters_RF = {
                    'n_estimators': [10,15,20,30],
                    'class_weight' :({1:10},'balanced'),
                    'max_features': ['auto', 'log2', None],

                    }



    gs_RF_clf = GridSearchCV(RF_clf, parameters_RF, n_jobs=-1)
    gs_RF_clf = gs_RF_clf.fit(learning_set, learning_lables)
    print 'RF fitted!'

    result['RF']=dict()
    result['RF']['best_score']=gs_RF_clf.best_score_
    result['RF']['best_parameter']=gs_RF_clf.best_params_
    result['RF']['label']=gs_RF_clf.predict(rest_set)
    result['RF']['score'] = gs_RF_clf.predict_proba(rest_set)[:,0]
    # print result['RF']['label'].tolist()
    # print result['RF']['score'].tolist()
    print result['RF']['best_score']
    print result['RF']['best_parameter']

    #Machine 4 KNN
    print 'Machine 4 KNN'
    knn_clf =  KNeighborsClassifier()


    parameters_knn = {
                    'n_neighbors': [3,5],
                    'weights': ['uniform','distance'],
                    'algorithm': ['auto'],
                    'leaf_size':[10,30,50]

                      }



    gs_knn_clf = GridSearchCV(knn_clf, parameters_knn, n_jobs=-1)
    gs_knn_clf = gs_knn_clf.fit(learning_set, learning_lables)
    print 'KNN fitted!'


    result['knn']=dict()
    result['knn']['best_score']=gs_knn_clf.best_score_
    result['knn']['best_parameter']=gs_knn_clf.best_params_
    result['knn']['label']=gs_knn_clf.predict(rest_set)
    result['knn']['score'] = gs_knn_clf.predict_proba(rest_set)[:, 0]
    # print result['knn']['label'].tolist()
    # print result['knn']['score'].tolist()
    print result['knn']['best_score']
    print result['knn']['best_parameter']

    print 'SGD'
    C = 1
    sample_weight = np.ones(learning_set.shape[0]) * C
    params_d = {"alpha": 10.0 ** -np.arange(1, 7)}

    sgd = SGDClassifier(class_weight={1: 5}, random_state=42,loss='log')
    clf_sgd = GridSearchCV(sgd, params_d, scoring='f1', fit_params={'sample_weight': sample_weight})
    clf_sgd = clf_sgd.fit(learning_set, learning_lables)

    result['sgd'] = dict()
    result['sgd']['best_score'] = clf_sgd.best_score_
    result['sgd']['best_parameter'] = clf_sgd.best_params_
    result['sgd']['label'] = clf_sgd.predict(rest_set)
    result['sgd']['score'] = clf_sgd.predict_proba(rest_set)[:, 0]
    # print result['knn']['label'].tolist()
    # print result['knn']['score'].tolist()
    print result['sgd']['best_score']
    print result['sgd']['best_parameter']

    #Machine 5 GradientBoosting
    # print 'Machine 5 GradientBoosting'
    # enc = LabelEncoder()
    # label_encoder = enc.fit(learning_lables)
    # learning_lables = label_encoder.transform(learning_lables)
    # vect=TfidfVectorizer(learning_set)
    # learn= (vect.fit_transform(learning_set)).toarray()
    # print learn.shape
    # param_test1 = {'n_estimators': range(30, 81, 10)}
    # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8,
    #                         max_features='sqrt', subsample=0.8, random_state=10),
    #                         param_grid=param_test1, n_jobs=-1)
    #
    # gsearch1.fit(learn, learning_lables)
    # print 'done1'
    #
    # param_test2 = {'max_depth': range(7, 16, 2), 'min_samples_split': range(400, 1201, 200),'min_samples_leaf': range(40, 71, 10)}
    #
    # gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=int(gsearch1.best_params_['n_estimators']), max_features='sqrt', subsample=0.8,
    #                         random_state=10),
    #                         param_grid=param_test2, n_jobs=-1)
    #
    # gsearch2.fit(learn, learning_lables)
    # print 'done2'
    #
    # param_test4 = {'max_features': range(7, 20, 2)}
    #
    # gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],
    #                         max_depth=gsearch2.best_params_['max_depth'], min_samples_split=gsearch2.best_params_['min_samples_split'],
    #                         min_samples_leaf=gsearch2.best_params_['min_samples_leaf'], subsample=0.8, random_state=10),
    #                         param_grid=param_test4, n_jobs=-1)
    #
    # gsearch3.fit(learn, learning_lables)
    # print gsearch3.best_params_
    # print 'done3'
    #
    #
    #
    #
    # GB_clf = Pipeline([('vect', TfidfVectorizer()),
    #                    ('dens',Densifier()),
    #                     ('clf', GradientBoostingClassifier()),
    #                     ])
    #
    # parameters_GB = {'vect__ngram_range': [(1, 1), (1, 2)],
    #                 'vect__max_df':[0.9,0.8,1.0],
    #                 'vect__min_df':[0.1,0.2,1],
    #                  'clf__subsample': [0.6, 0.8,1.0],
    #                  'clf__learning_rate':[0.1],
    #                  'clf__n_estimators':[gsearch1.best_params_['n_estimators']],
    #                  'clf__max_depth':[gsearch2.best_params_['max_depth']],
    #                  'clf__min_samples_split':[gsearch2.best_params_['min_samples_split']],
    #                  'clf__min_samples_leaf':[gsearch2.best_params_['min_samples_leaf']],
    #                  'clf__max_features':[gsearch3.best_params_['max_features']],
    #                  'clf__random_state':[10]
    #
    #                   }
    #
    # gs_GB_clf = GridSearchCV(GB_clf, parameters_GB, n_jobs=-1)
    # gs_GB_clf = gs_GB_clf.fit(learning_set, learning_lables)
    #
    # print 'GradientBoosting fitted!'
    # result['GB'] = dict()
    # result['GB']['best_score'] = gs_GB_clf.best_score_
    # result['GB']['best_parameter'] = gs_GB_clf.best_params_
    # result['GB']['label'] = gs_GB_clf.predict(rest_set)
    # result['GB']['score'] = gs_GB_clf.predict_proba(rest_set)[:, 0]
    # print result['GB']['best_score']
    # print result['GB']['best_parameter']


    return result


def getmachinesScore(result,i):
    lab=np.zeros(shape=result['knn']['label'].shape,dtype=np.int64)
    score=np.zeros(shape=result['knn']['score'].shape,dtype=np.float64)
    for item in result:


        lab=lab+(result[item]['label']).astype(np.int64)
        score=score+result[item]['score']


    score = score / len(result)#average scores!

    result['labelAll']=dict()
    for j in range(1,len(result)):
        label = [1 if t >= j else 0 for t in lab]# if atleast j machine say yes
        result['labelAll']['label'+str(j)]=label



    result['scoreAll'] = dict()
    for j in [0.55,0.6,0.65,0.7,0.75]:
        label = [0 if t >= j else 1 for t in score]  # if prob average more than j,  say yes
        result['scoreAll']['score' + str(j)] = label



    return result,score


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

def _get_baseline_SGD(sample_weight):
    params_d = {"alpha": 10.0**-np.arange(1,7)}

    sgd = SGDClassifier(class_weight={1:10}, random_state=42)

    clf = GridSearchCV(sgd, params_d, scoring='f1',fit_params={'sample_weight': sample_weight})

    return clf
def oneSGDmachine():


        texts, labels, pmids = _load_data('../data/proton-beam-merged.csv')
        labels = []
        getcrowdvotequestion1=Cmain()# change the label with first question label!
        for item in pmids:
            labels.append(getcrowdvotequestion1[item])

        vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000)
        X = vectorizer.fit_transform(texts)
        X = X.toarray()
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


        C = 1
        kf = KFold(n_splits=10, random_state=10)
        print X.shape[0]
        cm = np.zeros(len(np.unique(y)) ** 2)
        for train, test in kf.split(X_train):
            print test.shape[0]

            sample_weight = np.ones(X[train].shape[0]) * C
            clf = _get_baseline_SGD(sample_weight)

            clf.fit(X_train[train], y_train[train])
            print clf.best_score_
            print clf.best_params_

            y_pred = clf.predict(X_train[test])
            cm += confusion_matrix(y_train[test], y_pred).flatten()
        num_fold = kf.get_n_splits(X)
        print cm
        sensitivity, specificity, precision, loss = compute_measures(*cm / float(num_fold))

        print "\n----"
        sensitivity, specificity, precision, loss
        print "sensitivity: %s" % sensitivity
        print "specificity: %s" % specificity
        print "precision: %s" % precision
        print "loss: %s" % loss


        print "----"
        sample_weight = np.ones(X_train.shape[0]) * C
        clf = _get_baseline_SGD(sample_weight)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred).flatten()

        sensitivity, specificity, precision, loss = compute_measures(*cm)

        print "\n----"

        print "sensitivity: %s" % sensitivity
        print "specificity: %s" % specificity
        print "precision: %s" % precision
        print "loss: %s" % loss
        print "----"

        return sensitivity, specificity, precision, loss


def maketraingset():
    texts, labels, pmids = _load_data('../data/proton-beam-merged.csv')

    labels = []
    getcrowdvotequestion1 = Cmain()  # change the label with first question label!
    for item in pmids:
        labels.append(getcrowdvotequestion1[item])

    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000)
    X = vectorizer.fit_transform(texts)

    X = X.toarray()
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    return X,pmids,X_train,y_train

def makemachines(X_train,y_train):




    #*****************************************************************
    print 'sgd'
    C = 1
    sample_weight = np.ones(X_train.shape[0]) * C
    params_d = {"alpha": 10.0 ** -np.arange(1, 7)}
    sgd = SGDClassifier(class_weight={1: 10}, random_state=42)
    clfsgd = GridSearchCV(sgd, params_d, scoring='f1', fit_params={'sample_weight': sample_weight})
    clfsgd = clfsgd.fit(X_train, y_train)
    #*****************************************************************
    print 'svm'
    SVM_clf = SVC(class_weight={1: 10})
    parameters_SVM ={
                        'kernel': ['rbf'],
                        'probability':[True],
                        'C':[10,1000],
                        'gamma':['auto']

                        }

    gs_SVM_clf = GridSearchCV(SVM_clf, parameters_SVM, n_jobs=-1)
    gs_SVM_clf = gs_SVM_clf.fit(X_train, y_train)
    #*****************************************************************
    print 'rf'
    RF_clf =  RandomForestClassifier(class_weight={1: 10})
    parameters_RF = {
                    'n_estimators': [10,20,30],
                    'max_features': ['auto', None],

                    }



    gs_RF_clf = GridSearchCV(RF_clf, parameters_RF, n_jobs=-1)
    gs_RF_clf = gs_RF_clf.fit(X_train, y_train)

    # *****************************************************************
    print 'knn'
    knn_clf =  KNeighborsClassifier()
    parameters_knn = {
                    'n_neighbors': [3,5],
                    'weights': ['uniform','distance'],
                    'algorithm': ['auto'],
                    'leaf_size':[10,30,50]

                      }

    gs_knn_clf = GridSearchCV(knn_clf, parameters_knn, n_jobs=-1)
    gs_knn_clf = gs_knn_clf.fit(X_train, y_train)

    # *****************************************************************
    print 'nb'
    # NaiveBase_clf = MultinomialNB()
    #
    # parameters_NaiveBase = {
    #     'alpha': [1.0, 0.1, 0.01, 0.001],
    #     'fit_prior': (True, False)
    #
    # }
    #
    # gs_NaiveBase_clf = GridSearchCV(NaiveBase_clf, parameters_NaiveBase, n_jobs=-1)
    # gs_NaiveBase_clf = gs_NaiveBase_clf.fit(X_train, y_train)

    print 'finish'

    return clfsgd, gs_SVM_clf,gs_knn_clf,gs_RF_clf










