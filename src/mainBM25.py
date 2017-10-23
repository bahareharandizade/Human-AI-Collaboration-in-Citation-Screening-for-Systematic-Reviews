from parse import *
from query import QueryProcessor
import operator
import pandas as pd
from Initialization import *



Orpapers=dict()
Orderdataset=pd.DataFrame()
feature_dict = {i: label for i, label in zip(
            range(6),
            ('id',
             'title',
             'abstract',
             'pid',
             'year',
             'lable'))}


def makeCorpus(dataset):

    #
    # with open('../data/ProtonbeamTitle.csv') as f:
    #     lines = ''.join(f.readlines())
    #
    # itemsT = [x.rstrip() for x in lines.split('\n')[:-1]]
    # with open('../data/ProtonbeamAbstract.csv') as f:
    #     lines = ''.join(f.readlines())
    # itemsA = [x.rstrip() for x in lines.split('\n')[:-1]]

    final = list()
    title = list()
    abs = list()

    for item in dataset[[1]]:

        if item[0] == '[':
            title.append(item[1:-4])
        elif item[0] == '"':
            title.append(item[1:-4])

        else:
            title.append(item[0:-3])


    for item in dataset[[2]]:
        abs.append(item[1:-3])

    for (item1, item2) in zip(title, abs):
        final.append(item1 + ' ' + item2)

    with open('../data/CorpusProtonbeam.txt', 'w') as f:

        i = 1
        for item in final:
            f.write('\n' + "#" + str(i) + '\n')
            f.write(item)
            i = i + 1




def rankpapers():
    qp = QueryParser(filename='../data/queriesProtonbeam.txt')
    cp = CorpusParser(filename='../data/CorpusProtonbeam.txt')
    qp.parse()
    queries = qp.get_queries()

    cp.parse()
    corpus = cp.get_corpus()



    proc = QueryProcessor(queries, corpus)
    results = proc.run()
    qid = 0
    for result in results:
        sorted_x = sorted(result.iteritems(), key=operator.itemgetter(1))
        sorted_x.reverse()

        index = 0
        # maxScore=sorted_x[0][1]
        for i in sorted_x:
            tmp = (qid, i[0], index, i[1])
            Orpapers[i[0]] = i[1]
            print '{:>1}\tQ0\t{:>4}\t{:>2}\t{:>12}\tNH-BM25'.format(*tmp)
            index += 1
        qid += 1


def writerankpapers(dataset):

    pid=dataset[[3]]
    for item in Orpapers:
        row=dataset.loc[dataset['pid'] == item]
        Orderdataset.append(row,ignore_index=True)
        pid.remove(item)
    for item in pid:
        row = dataset.loc[dataset['pid'] == item]
        Orderdataset.append(row, ignore_index=True)
        Orpapers[item]=0.0
    Orderdataset.columns = [l for i, l in sorted(feature_dict.items())]
    return Orderdataset,Orpapers
    # with open('../data/protonbeam.csv') as f:
    #     lines = ''.join(f.readlines())
    #
    # items = [x.rstrip() for x in lines.split('\n')[:-1]]
    # items.append('4749,Proton beam radiotherapy of iris melanoma.,"PURPOSE: To report on outcomes after proton beam radiotherapy of iris melanoma.<br />METHODS AND MATERIALS: Between 1993 and 2004, 88 patients with iris melanoma received proton beam radiotherapy, with 53.1 Gy in 4 fractions.<br />RESULTS: The patients had a mean age of 52 years and a median follow-up of 2.7 years. The tumors had a median diameter of 4.3 mm, involving more than 2 clock hours of iris in 32% of patients and more than 2 hours of angle in 27%. The ciliary body was involved in 20%. Cataract was present in 13 patients before treatment and subsequently developed in another 18. Cataract had a 4-year rate of 63% and by Cox analysis was related to age (p = 0.05), initial visual loss (p < 0.0001), iris involvement (p < 0.0001), and tumor thickness (p < 0.0001). Glaucoma was present before treatment in 13 patients and developed after treatment in another 3. Three eyes were enucleated, all because of recurrence, which had an actuarial 4-year rate of 3.3% (95% CI 0-8.0%).<br />CONCLUSIONS: Proton beam radiotherapy of iris melanoma is well tolerated, the main problems being radiation-cataract, which was treatable, and preexisting glaucoma, which in several patients was difficult to control.",16111578,2005,1,1')
    # for t in items:
    #     x=t.index(',')
    # Torder[t[0:x]] = t[x:]



    # with open('../data/OrderProtonbeamBM25.csv','w') as e:
    #     i=1
    #     for item in Orpapers.keys():
    #         e.write(str(i)+','+Torder[item]+'\n')
    #         del Torder[item]
    #         i=i+1
    #     for key in Torder.keys():
    #         e.write(str(i)+','+Torder[key]+'\n')
    #         i=i+1



def BM25(dataset):
    makeCorpus(dataset)
    rankpapers()
    return writerankpapers(dataset)