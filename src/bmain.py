import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from mainBM25 import BM25
from Crowd import *
from Initialization import *
from vectorizedocs import *
from machineAPI import *


if __name__ == '__main__':

    class_names=["Out", "In"]
    Mode={1:"1",2:"2",3:"3"}


    def write2file(result, y_test, filename):

        output = pd.DataFrame(data={"real": y_test, "sentiment": result})
        output.to_csv(os.path.join(os.path.dirname(__file__), 'data', ''.join([filename, '.csv'])), index=False, quoting=3)


    def plot_confusion_matrix(y_test,result, classes,
                              normalize=True,
                              title='Confusion matrix',
                              cmap=plt.cm.binary):

        cm = confusion_matrix(y_test, result)
        np.set_printoptions(precision=2)

        res=plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=len(y_test))
        plt.title(title)
        plt.colorbar(res)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

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
                     color="black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')




    #framework start from here

    #we can put K and M to loop as well

    fig_path = '../plots/{0}-Run:{1}-K:{2}-M:{3}.png'
    K=500
    framework=InitialExperimant("1",K)# start machine with k=500 papers
    framework.Random()#initialize for random
    #loop
    vectorizer=vectorize()
    M=50 # add in each iteration

    for i in range(0,5):

        learning,rest=vectorizer.TfidfVectorizer(framework.get_learning_set()[[2]].values,
                                            framework.get_restset()[[2]].values,
                                            tokenizer=vectorizer.tokenize_only,
                                            ngram_range_min=1,
                                            ngram_range_max=3,
                                            stop_words='english',
                                            lowercase=True,
                                            max_df=1.0,
                                            min_df=1,
                                            max_features=200000)



        print learning.shape
        print rest.shape
        #lsa running for dimension reduction
        svd = TruncatedSVD(500)# the new feature dimension can be set here!
        lsa = make_pipeline(svd, Normalizer(copy=False))
        learning_lsa = lsa.fit_transform(learning)
        rest_lsa = lsa.transform(rest)
        #
        learning_labels=[val for sublist in framework.learningset[[5]].values for val in sublist]#convert list of list to list
        # print learning_labels
        machine_run(learning_lsa,learning_labels,rest_lsa,i,K,M)
        combinescore(i,K,M)
        y_pred,meanscore=getmachinesScore(i,K,M)
        indexsort = np.argsort(meanscore)#get index of sort array
        # print type(meanscore)


        y_real=[val for sublist in framework.restset[[5]].values for val in sublist]
        # print y_pred
        # print y_real
        plot_confusion_matrix(y_real, y_pred, classes=class_names, normalize=True,
                              title='Confusion matrix Run:'+str(i)+' K:'+str(K)+'M:'+str(M))

        name = fig_path.format('random',i,K,M)
        plt.savefig(name)
        plt.clf()

        # select the worst one--> near to hyperplane--> in the middle of list
        msample = indexsort[(len(indexsort) / 2) - (M / 2):(len(indexsort) / 2) + (M / 2)]  # get the middle as sample

        columns = [l for i, l in sorted(framework.feature_dict.items())]
        Mdataframe = pd.DataFrame(columns=columns)#make a frame for M sample
        restpid=framework.get_restset()['pid'].values
        for i in msample:
            item = restpid[i]
            row = framework.dataset.loc[framework.dataset['pid'] == str(item)]
            Mdataframe=pd.concat([Mdataframe, row], axis=0, ignore_index=True)


        Mdataframelabled=askCrowd(Mdataframe)#ask crowd to label M items

        framework.learningset.append(Mdataframelabled,ignore_index=True)# add these items to learning set

        framework.restset=framework.restset[~framework.restset['pid'].isin(Mdataframelabled['pid'])]#delete M items from restset






















