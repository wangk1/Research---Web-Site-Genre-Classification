from sklearn.pipeline import Pipeline

__author__ = 'Kevin'

from matplotlib.backends.backend_pdf import PdfPages
from data.util import unpickle_obj,pickle_obj
from data.training_testing import Training,Testing
from data import LearningSettings
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from unsupervised.clustering import Clustering
from unsupervised.graph_cut.graph_cut import *
from analytics.graphics import *
from sklearn.feature_selection import SelectKBest,chi2
from analytics.graphics import subplot_four_corner

from util.base_util import normalize_genre_string
import os





pickle_dir="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir"
def unsupervised(label,train_set,clusterer,clustering_alg):
    print("Unsupervised Algorithm training size: {}".format(train_set.X.shape))

    for num_cluster in sorted(label.num_clusters,reverse=True):

        X,y,ref_ids=train_set.to_matrices()
        nn=clustering_alg(n_clusters=num_cluster)

        #nn=SpectralClustering(n_clusters=num_cluster)
        print("Using {}".format(str(nn)))

        res_labels=nn.fit_predict(X)

        occurence_dict=Clustering().get_clusters_genre_distribution(y,res_labels)

        res_file="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\unsupervised\\{}\\{}\\{} clusters.pdf"\
            .format(label.clustering_alg,label.res_dir,len(occurence_dict))

        res_dir="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\unsupervised\\{}\\{}" \
                .format(label.clustering_alg,label.res_dir)

        os.makedirs(res_dir,exist_ok=True)

        res_path=os.path.join(pickle_dir,label.clustering_alg+(label.res_dir if label.res_dir else "_{}".format(label.res_dir)))
        pickle_obj(res_labels,res_path)
        print("Finished pickling the {} results to {}".format(label.clustering_alg,res_path))

        with PdfPages(res_file) as pdf:
            plt_num=0
            save_fig=False
            figure=None
            for cluster_num,cluster_genre_freq in occurence_dict.items():
                if len(cluster_genre_freq) < 2 or (len(cluster_genre_freq)<3 and all(map(lambda x:x[1]==1,cluster_genre_freq.items()))):
                    continue
                save_fig=True
                last_plt,figure=subplot_four_corner(plt_num)

                #plot it
                file_name="{}_genre_distribution".format(cluster_num)+".pdf"
                #axis=plt.subplot(1,1,plt_num)
                num_samples=np.sum(res_labels==cluster_num)
                print("Total number of samples in cluster {} is {}".format(cluster_num,num_samples))

                plot_word_frequency("cluster {}, num samples: {}".format(cluster_num,num_samples),
                                    cluster_genre_freq)

                plt_num+=1
                if last_plt:
                    save_fig=False
                    pdf.savefig(figure)
                    plt.close()

            if figure is not None and save_fig:
                pdf.savefig(figure)
                plt.close()

        inter_cluster,inter_cluster_count,intra_cluster,intra_cluster_count=Clustering().cluster_closeness(nn.cluster_centers_,X,res_labels)



        print("\n{} {} clusters:".format(label.clustering_alg,num_cluster))
        print("Avg Inter: {}".format([c[1]/(t[1] == 0 or t[1]) for c,t in zip(inter_cluster
                                                               ,inter_cluster_count)]))
        print("Inter count: {}".format([c[1] for c in inter_cluster_count]))
        print("Avg Intra: {}".format([c[1]/(t[1] == 0 or t[1]) for c,t in zip(intra_cluster
                                                               ,intra_cluster_count)]))
        print("Intra count: {}".format([c[1] for c in intra_cluster_count]))

def cluster_analysis(res_dir):
    pass


if __name__=="__main__":
    mapping={"short_genres":"short_genre","index":"ref_index","bow":"attr_map"}

    #s=SourceMapper(URLBow.objects(),mapping)
    X_pickle_path=os.path.join(pickle_dir,"X_summary_pickle")
    y_pickle_path=os.path.join(pickle_dir,"y_summary_pickle")
    ref_index_pickle_path=os.path.join(pickle_dir,"refIndex_summary_pickle")

    mapping={"short_genres":"short_genre","index":"ref_index","bow":"attr_map"}

    #SETTING UP LABEL
    label=LearningSettings(type="unsupervised",dim_reduction="chi",feature_selection="summary",num_feats=10000)
    label.res_dir="cluster_9"
    #label.res_dir=""

    label.clustering_alg="kNN"
    clustering_alg=KMeans
    label.num_clusters=list({20,18,16,14,12,10,9,8,7,6,5,4,3})
    #label.num_clusters=list({9})



    #generate_random_sample(unpickle_obj(X_pickle_path),unpickle_obj(y_pickle_path),unpickle_obj(ref_index_pickle_path),50000)

    #LOAD DATA
    train_set=Training(label,pickle_dir=pickle_dir)
    train_set.load_training()

    #FEATURE SELECTION
    best_k_attr=10000
    feature_selector=Pipeline([("chi2",SelectKBest(chi2,best_k_attr))])

    print("Choosing best {} features".format(best_k_attr))

    print("Normalizing to LV1")
    #NORMALIZING THE Y
    train_set.y=[[normalize_genre_string(g,1) for g in r] for r in (row for row in train_set.y)]

    clusterer=Clustering()
    clusterer.feature_selection(train_set,feature_selector,fit=True)

    #MODIFY X AND Y
    #get subcluster 0

    cluster_assign=unpickle_obj(os.path.join(pickle_dir,label.clustering_alg))
    selector=(cluster_assign==0)


    train_set.X=train_set.X[selector]
    train_set.y=np.array(train_set.y)[selector]
    train_set.ref_indexes=train_set.ref_indexes[selector]


    unsupervised(train_set=train_set,label=label,clusterer=clusterer,clustering_alg=clustering_alg)
