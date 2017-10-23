from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def Esvc(learningset,label,restset,kernel="rbf", class_weight="balanced",C=1,gamma='auto'):
    if class_weight=='weighted':
        class_weight={1:10}
    if gamma!='auto':
        gamma=float(gamma)
    if kernel=='linear' and gamma=='0':
        gamma='auto'
    SVM = svm.SVC(kernel=kernel,
                         probability=True,
                         C=float(C),
                         class_weight=class_weight,
                         gamma=gamma)

    SVM.fit(learningset, label)
    score=SVM.predict_proba(restset)
    restlabel=SVM.predict(restset)
    print restlabel
    print score
    print SVM.classes_

    return score[:,0],restlabel


def randomforrest(learningset, label, restset,n_estimators=10, max_features='auto',class_weight="balanced"):
    if class_weight=='weighted':
        class_weight={1:10}
    randforrest=RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_features=max_features,
                                       class_weight=class_weight)

    randforrest.fit(learningset,label)
    score=randforrest.predict_proba(restset)
    restlabel=randforrest.predict(restset)
    return score[:,0],restlabel

#add more classifier here!