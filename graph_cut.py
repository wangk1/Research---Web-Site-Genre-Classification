__author__ = 'Kevin'
from sklearn.pipeline import Pipeline

import os
from data.training_testing import Training
from unsupervised.graph_cut.graph_cut import *
from classification_attribute.feature_selection import chi_squared_feature_select


from util.base_util import normalize_genre_string


pickle_dir="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir"

if __name__=="__main__":
    mapping={"short_genres":"short_genre","index":"ref_index","bow":"attr_map"}

    #s=SourceMapper(URLBow.objects(),mapping)
    X_pickle_path=os.path.join(pickle_dir,"X_summary_pickle")
    y_pickle_path=os.path.join(pickle_dir,"y_summary_pickle")
    ref_index_pickle_path=os.path.join(pickle_dir,"refIndex_summary_pickle")

    mapping={"short_genres":"short_genre","index":"ref_index","bow":"attr_map"}


    label="summary_unsupervised_chi_top1cls_10000"

    #generate_random_sample(unpickle_obj(X_pickle_path),unpickle_obj(y_pickle_path),unpickle_obj(ref_index_pickle_path),1000)

    #load training, feature selection
    train_set=Training(label,pickle_dir=pickle_dir)
    train_set.load_training()
    train_set.y=np.array([list(set(normalize_genre_string(genre,1) for genre in g_list)) for g_list in train_set.y])
    train_set.X=chi_squared_feature_select(train_set.X,train_set.y,k_best=10000)

    params=GraphCutParams(X=train_set.X,y=train_set.y,ref_id=train_set._ref_indexes,
                   k_closest_neighbors=4,vocab_size=train_set._X.shape[1],num_clusters=3)

    alpha_beta_swap(params)