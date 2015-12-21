from matplotlib.backends.backend_pdf import PdfPages

__author__ = 'Kevin'


import os,operator as op
import collections as coll
import numpy as np
import matplotlib.pyplot as plt
from analytics.graphics import subplot_four_corner,plot_word_frequency

import sklearn.metrics.pairwise as pw

from util.Logger import Logger
from data import LearningSettings

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

    def generate_cluster_distribution_graphs(self,res_file,occurence_dict,res_labels):
        """
        Generate graphics of each cluster and the genre distribution from the occurence dictionary of genre in each
            cluste, res_labels for the training set, and save the graph to the res_file

        :param res_file:
        :param occurence_dict:
        :param res_labels:
        :return:
        """
        with PdfPages(res_file) as pdf:
            plt_num=0
            save_fig=False
            figure=None
            for cluster_name,cluster_genre_freq in occurence_dict.items():

                save_fig=True
                last_plt,figure=subplot_four_corner(plt_num)

                #axis=plt.subplot(1,1,plt_num)
                num_samples=np.sum(res_labels==cluster_name)
                print("Total number of samples in cluster {} is {}".format(cluster_name,num_samples))

                plot_word_frequency("cluster {}, num samples: {}".format(cluster_name,num_samples),
                                    cluster_genre_freq)
                plt_num+=1
                if last_plt:
                    save_fig=False
                    pdf.savefig(figure)
                    plt.close()

            if figure is not None and save_fig:
                pdf.savefig(figure)
                plt.close()
