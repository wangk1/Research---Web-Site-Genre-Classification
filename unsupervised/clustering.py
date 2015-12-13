__author__ = 'Kevin'


import os,operator as op
import collections as coll


import sklearn.metrics.pairwise as pw

from util.Logger import Logger
from data import Label

clustering_logger=Logger()

class Clustering:
    """
    Important formats:
        Coming soon

    Class used as a pipeline for Clustering. Makes things easier to manage

    An instance of this class is used to keep track of the state of classification

    1. Be able to save the fitted attribute vectorizer to disk to conserve memory and avoid reloading vocab each time
    2. Likewise for training and testing samples
    3. Keep track of the outputting location of the classification result files

    """

    def get_clusters_genre_distribution(self,y,pred_y):
        pred_to_actual=coll.defaultdict(lambda: coll.Counter())

        for p_y,a_y in zip(pred_y,y):
            pred_to_actual[p_y].update(a_y)

        return pred_to_actual

    def cluster_closeness(self,centers,X,pred_y,score_func=pw.pairwise_distances,min_cluster_size=2):
        """
        Calculate how "close" each cluster is to the webpages within the cluster and also to the respective outside

        :param X:
        :param y:
        :param pred_y:
        :return:
        """

        inter_cluster=coll.defaultdict(lambda:0)
        inter_cluster_count=coll.defaultdict(lambda:0)
        intra_cluster=coll.defaultdict(lambda:0)
        intra_cluster_count=coll.defaultdict(lambda:0)

        for c,p_y in enumerate(pred_y):
            #compare the webpage to its cluster center
            inter_diffs=[score_func(X[c],center)[0,0] for center in centers]

            for cluster,diff in enumerate(inter_diffs):
                if cluster==p_y:
                    intra_cluster_count[p_y]+=1
                    intra_cluster[p_y]+=diff
                else:
                    inter_cluster[cluster]+=diff
                    inter_cluster_count[cluster]+=1

        #filter out the clusters that are too small
        intra_cluster_count=sorted(intra_cluster_count.items(),key=op.itemgetter(0))
        intra_cluster_count=[c for c in filter(lambda x: x[1]>min_cluster_size,intra_cluster_count)]
        cluster_num_selection=set((i[0] for i in intra_cluster_count))
        intra_cluster=list(filter(lambda x: x[0] in cluster_num_selection,sorted(intra_cluster.items(),key=op.itemgetter(0))))

        inter_cluster=list(filter(lambda x: x[0] in cluster_num_selection,sorted(inter_cluster.items(),key=op.itemgetter(0))))
        inter_cluster_count=list(filter(lambda x: x[0] in cluster_num_selection,sorted(inter_cluster_count.items(),key=op.itemgetter(0))))


        return inter_cluster,inter_cluster_count,intra_cluster,intra_cluster_count

    def feature_selection(self,data_set,feature_selector,fit=True):
        """
        Perform feature selection. Must be done before loading testing sets

        :param feat_selector:
        :return:
        """

        assert hasattr(feature_selector,"transform")

        clustering_logger.info("Pre feature selection: num features: {}".format(data_set.X.shape[1]))

        if fit:
            feature_selector.fit(data_set.X,data_set.y)

        train_X=feature_selector.transform(data_set.X)

        clustering_logger.info("Post feature selection: num features: {}".format(train_X.shape[1]))

        data_set.X=train_X



