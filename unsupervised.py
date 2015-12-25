from sklearn.pipeline import Pipeline

__author__ = 'Kevin'

from data.util import unpickle_obj,pickle_obj
from data.training_testing import Training,Testing
from data import LearningSettings
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from unsupervised.clustering import Clustering
from unsupervised.graph_cut.graph_cut import *
from sklearn.feature_selection import SelectKBest,chi2
from util.Logger import Logger

from util.base_util import normalize_genre_string
import os

PICKLE_DIR="pickle_dir"
UNSUPERVISED_DIR="classification_res\\unsupervised"

def unsupervised(settings,train_set,clusterer,clustering_alg_cls):
    clustering_logger.info("Unsupervised Algorithm training size: {}".format(train_set.X.shape))

    for num_cluster in sorted(settings.num_clusters,reverse=True):

        X,y,ref_ids=train_set.to_matrices()

        additional_notes=""
        """
        if train_set.X.shape[0]<=settings.spectre_clustering_limit:
            clustering_alg=SpectralClustering(n_clusters=num_cluster)
            additional_notes="_spectral"
        else
        """
        clustering_alg=clustering_alg_cls(n_clusters=num_cluster)

        clustering_logger.info("Using {}".format(str(clustering_alg)+additional_notes))

        res_labels=clustering_alg.fit_predict(X)

        occurence_dict=clusterer.get_clusters_genre_distribution(y,res_labels)

        #the directory to store the results of clustering
        res_dir=os.path.join(UNSUPERVISED_DIR,settings.clustering_alg,*settings.parent_clusters)


        #ELIMATE CLUSTER LESS THAN 2 pages in size
        for cluster_name, cluster_genre_count in list(occurence_dict.items()):
            total_count_in_cluster=sum((count for genre,count in cluster_genre_count.items()))

            if total_count_in_cluster < 12:
                del occurence_dict[cluster_name]
            else:
                path=os.path.join(res_dir,"{}_{}_pages".format(num_cluster,cluster_name))
                #OUTPUT the pages in the current cluster
                clusterer.output_pages_in_cluster(path,train_set.ref_indexes[res_labels==cluster_name])


        res_file="{}/{}.pdf".format(res_dir,str(num_cluster))

        os.makedirs(res_dir,exist_ok=True)

        clusterer.generate_cluster_distribution_graphs(res_file,occurence_dict,res_labels)

        #output closeness metrics
        if additional_notes=="":
            inter_cluster,inter_cluster_count,intra_cluster,intra_cluster_count=Clustering().cluster_closeness(clustering_alg.cluster_centers_,X,res_labels)
            clusterer.output_cluster_closeness("{}/{}.txt".format(res_dir,num_cluster),inter_cluster,
                                               inter_cluster_count,intra_cluster,intra_cluster_count)

        #do a dfs on clusters bigger than the prescribed size
        if settings.break_up_clusters:
            breakup_candidate=[]

            for i in range(0,num_cluster):
                if np.sum(res_labels==i)>=settings.max_cluster_size:
                    breakup_candidate.append(i)

            X_path=os.path.join(res_dir,"X")
            y_path=os.path.join(res_dir,"y")
            ref_indexes_path=os.path.join(res_dir,"ref_indexes")

            clustering_logger.info("Pickling X,y,ref_index to conserve memory")
            pickle_obj(train_set.X,X_path)
            pickle_obj(train_set.y,y_path)
            pickle_obj(train_set.ref_indexes,ref_indexes_path)

            for cluster_name in breakup_candidate:
                clustering_logger.info("Breaking up cluster {} of size greater than {}".format(cluster_name,settings.max_cluster_size))

                settings.parent_clusters.append("{}_{}".format(num_cluster,cluster_name))

                selector=(res_labels==cluster_name)

                train_set.X=train_set.X[selector]
                train_set.y=train_set.y[selector]
                train_set.ref_indexes=train_set.ref_indexes[selector]

                unsupervised(settings,train_set,clusterer,clustering_alg_cls)

                settings.parent_clusters.pop()

                train_set.X=unpickle_obj(X_path)
                train_set.y=unpickle_obj(y_path)
                train_set.ref_indexes=unpickle_obj(ref_indexes_path)

            #remove the cache files
            os.remove(ref_indexes_path)
            os.remove(X_path)
            os.remove(y_path)


if __name__=="__main__":
    clustering_logger=Logger()
    """
    Unsupervised Clustering bootstrap
    """

    mapping={"short_genres":"short_genre","index":"ref_index","bow":"attr_map"}

    #s=SourceMapper(URLBow.objects(),mapping)
    X_pickle_path=os.path.join(PICKLE_DIR,"X_summary_pickle")
    y_pickle_path=os.path.join(PICKLE_DIR,"y_summary_pickle")
    ref_index_pickle_path=os.path.join(PICKLE_DIR,"refIndex_summary_pickle")

    mapping={"short_genres":"short_genre","index":"ref_index","bow":"attr_map"}

    #SETTING UP LABEL
    settings=LearningSettings(type="unsupervised",dim_reduction="chi",feature_selection="summary",num_feats=10000)
    settings.parent_clusters=[] #used to record a tree of parent clusters for the current cluster

    settings.clustering_alg="kNN"
    clustering_alg=KMeans
    settings.num_clusters=list({10})
    settings.max_cluster_size=10000 #the cluster will be further broken up if it is greater than this size
    settings.break_up_clusters=True
    settings.spectre_clustering_limit=15000 # if the cluster is less than 15K in size, use spectre clustering instead

    #LOAD DATA
    #generate_random_sample(unpickle_obj(X_pickle_path),unpickle_obj(y_pickle_path),unpickle_obj(ref_index_pickle_path),50000)

    train_set=Training(settings,pickle_dir=PICKLE_DIR)
    train_set.load_training()

    #FEATURE SELECTION
    best_k_attr=10000
    feature_selector=Pipeline([("chi2",SelectKBest(chi2,best_k_attr))])

    clustering_logger.info("Choosing best {} features".format(best_k_attr))

    clustering_logger.debug("Normalizing to LV1")
    #NORMALIZING THE Y
    train_set.y=np.array([[normalize_genre_string(g,1) for g in r] for r in (row for row in train_set.y)])

    clusterer=Clustering()
    clusterer.feature_selection(train_set,feature_selector,fit=True)

    unsupervised(train_set=train_set, settings=settings,clusterer=clusterer, clustering_alg_cls=clustering_alg)
