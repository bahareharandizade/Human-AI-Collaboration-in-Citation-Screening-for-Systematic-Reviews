import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from Crowd import *
from Initialization import *
from vectorizedocs import *
from APImachine import *


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


    result = dict()
    for K in [20]:
        framework = InitialExperimant("1", K)  # start machine with k=100 papers --initialize for random
        framework.Balanced()
        result[K]=dict()

        for M in [10]:
            framework.learningset=framework.orginallearningset
            framework.restset=framework.orginalrestingset
            print len(framework.learningset)
            print len(framework.restset)
            result[K][M]=dict()


            for i in range(0,15):
                print len(framework.learningset)
                print len(framework.restset)
                print K
                print M
                print i

                learning_labels=[int(val) for sublist in framework.learningset[[2]].values for val in sublist]#convert list of list to list
                print learning_labels


                resultM=machineRun(framework.get_learning_set()[[1]].values,learning_labels,framework.get_restset()[[1]].values,vectorize.tokenize_only,i,K,M)
                resultagg,meanscore=getmachinesScore(resultM,i)
                result[K][M][str(i)] = dict()
                result[K][M][str(i)] = resultagg

                y_real=[val for sublist in framework.restset[[2]].values for val in sublist]
                result[K][M][str(i)]['real'] = dict()
                result[K][M][str(i)]['real']['reallabel']=y_real


                indexsort = np.argsort(meanscore)  # get index of sort array
                # select the worst one--> near to hyperplane--> in the middle of list
                msample = indexsort[(len(indexsort) / 2) - (M / 2):(len(indexsort) / 2) + (M / 2)]  # get the middle as sample

                columns = [l for i, l in sorted(framework.feature_dict.items())]
                Mdataframe = pd.DataFrame(columns=columns)#make a frame for M sample
                restpid=framework.get_restset()['pmid'].values
                for i in msample:
                    item = restpid[i]
                    row = framework.dataset.loc[framework.dataset['pmid'] == str(item)]
                    Mdataframe=pd.concat([Mdataframe, row], axis=0, ignore_index=True)


                Mdataframelabled=askCrowd(Mdataframe,'1')#ask crowd to label M items
                print len(Mdataframelabled)
                framework.learningset = framework.learningset.append(Mdataframelabled,ignore_index=True)# add these items to learning set

                framework.restset=framework.restset[~framework.restset['pmid'].isin(Mdataframelabled['pmid'])]#delete M items from restset


    p_file = '../data/{0}-K:{1}.pkl'
    filename = p_file.format('random',K)
    pickle.dump(result, open(filename, 'w'))




















