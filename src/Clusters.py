from sklearn.cluster import KMeans
import numpy as np
import pickle


class clusterDocs:


    def Kmeans(num_clusters,matrix,k):
        sample_per_cluster=k/num_clusters
        km = KMeans(n_clusters=num_clusters).fit(matrix)
        closest = []
        for i in num_clusters:
            d = km.transform(matrix)[:, i]
            closest.extend((np.argsort(d)[:-(sample_per_cluster + 1):-1]).tolist())

        return list(set(closest))

    def Clustermain(self,data,k):# the number of samples for learning set!
        results=[]

        for cluster in ['kmean']:
            results[cluster]=dict()
            for i in range(2,11):
                 km,samples=Kmeans(i,data,k)# i number of clusters
                 results[cluster][i]=samples

        pickle.dump(results, open('../data/clusters.pkl', 'w'))


