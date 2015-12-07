from sklearn.pipeline import Pipeline

__author__ = 'Kevin'

from matplotlib.backends.backend_pdf import PdfPages
from data.util import unpickle_obj,pickle_obj
from data.training_testing import Training
from sklearn.cluster import KMeans,SpectralClustering
from unsupervised.clustering import Clustering
from unsupervised.graph_cut.graph_cut import *
from analytics.graphics import *
import operator as op
from sklearn.decomposition import SparsePCA
from sklearn.feature_selection import SelectKBest,chi2
from analytics.graphics import subplot_four_corner

from util.base_util import normalize_genre_string

def pick_random_samples(X,y,ref_index,num):
    """
    Pickle random X,y,ref_index

    :param X:
    :param y:
    :param ref_index:
    :param num:
    :return:
    """
    choices=list(np.array(rand.sample(range(0,len(ref_index)),num)))

    return X[choices],y[choices],ref_index[choices]

def generate_random_sample(X,y,ref_index,num):

    selector=np.not_equal(ref_index,None)
    ref_index=ref_index[selector]
    X=X[selector]
    y=y[selector]

    sample_X_path=os.path.join(pickle_dir,"cluster_trainX_summary_pickle")
    sample_y_path=os.path.join(pickle_dir,"cluster_trainy_summary_pickle")
    sample_ref_index_path=os.path.join(pickle_dir,"cluster_trainRefIndex_summary_pickle")

    X,y,ref_index=pick_random_samples(X,y,ref_index,num)


    pickle_obj(X,sample_X_path)
    pickle_obj(y,sample_y_path)
    pickle_obj(ref_index,sample_ref_index_path)


pickle_dir="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir"

if __name__=="__main__":
    mapping={"short_genres":"short_genre","index":"ref_index","bow":"attr_map"}

    #s=SourceMapper(URLBow.objects(),mapping)
    X_pickle_path=os.path.join(pickle_dir,"X_summary_pickle")
    y_pickle_path=os.path.join(pickle_dir,"y_summary_pickle")
    ref_index_pickle_path=os.path.join(pickle_dir,"refIndex_summary_pickle")

    mapping={"short_genres":"short_genre","index":"ref_index","bow":"attr_map"}


    label="summary_unsupervised_chi_top1cls_10000"

    generate_random_sample(unpickle_obj(X_pickle_path),unpickle_obj(y_pickle_path),unpickle_obj(ref_index_pickle_path),10000)

    #load training
    train_set=Training(label,pickle_dir=pickle_dir)
    train_set.load_training()


    clusterer=Clustering()

    best_k_attr=5000
    #sparse pca to choose the 10k best components
    print("Choosing best {} features".format(best_k_attr))

    print("Normalizing to LV1")
    #normalize the y to 1st level
    train_set.y=[[normalize_genre_string(g,1) for g in r] for r in (row for row in train_set.y)]

    try:
        train_set.X=clusterer.feature_selection(train_set.X,train_set.y,SparsePCA(n_components=best_k_attr))
    except Exception as ex:
        print("SparsePCA failed, trying KBest with chisq: {}".format(str(ex)))
        train_set.X=clusterer.feature_selection(train_set.X,train_set.y,Pipeline([("chi2",SelectKBest(chi2,best_k_attr))]))


    #num_clusters={20,18,16,14,12,10,9,8,7,6,5,4,3}
    num_clusters={8,7,6,5,4,3}
    clustering_alg="kNN"
    clustering_alg="spectral"



    for num_cluster in sorted(num_clusters,reverse=True):

        X,y,ref_ids=train_set.to_matrices()


        #nn=KMeans(n_clusters=num_cluster)
        nn=SpectralClustering(n_clusters=num_cluster)
        print("Using {}".format(str(nn)))

        res_labels=nn.fit_predict(X)

        occurence_dict=Clustering().get_clusters_genre_distribution(y,res_labels)

        res_file="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\unsupervised\\{}\\{} clusters.pdf"\
            .format(clustering_alg,num_cluster)


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
                plot_word_frequency("{} clusters, cluster {} genre distribution".format(num_cluster,cluster_num),
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



        print("\n{} {} clusters:".format(clustering_alg,num_cluster))
        print("Avg Inter: {}".format([c[1]/(t[1] == 0 or t[1]) for c,t in zip(inter_cluster
                                                               ,inter_cluster_count)]))
        print("Inter count: {}".format([c[1] for c in inter_cluster_count]))
        print("Avg Intra: {}".format([c[1]/(t[1] == 0 or t[1]) for c,t in zip(intra_cluster
                                                               ,intra_cluster_count.items())]))
        print("Intra count: {}".format([c[1] for c in intra_cluster_count]))

    #generate_random_sample(unpickle_obj(X_pickle_path),unpickle_obj(y_pickle_path),(unpickle_obj(ref_index_pickle_path)),2000)

    # #take out the nones
    # X=unpickle_obj(X_pickle_path)
    # y=unpickle_obj(y_pickle_path)
    # index=unpickle_obj(ref_index_pickle_path)

    # selector=np.not_equal(index,None)
    # index=index[selector]
    # X=X[selector]
    # y=y[selector]
    #
    #
    # pickle_obj(X,X_pickle_path)
    # pickle_obj(y,y_pickle_path)
    # pickle_obj(index,ref_index_pickle_path)




