def compute_measures(tn, fp, fn, tp):
    R = 10
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)
    print 'tp'+str(tp)
    print 'fp'+str(fp)
    print 'fn'+str(fn)
    print 'tn'+str(tn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision  = tp / (tp + fp)
    loss = (fp+(R*fn))/ (tp+fp+fn+tn)
    #man = fp / (tp+fp)
    return sensitivity, specificity, precision, loss
    #return sensitivity, man