import itertools
import numpy as np
from classification.classifiers import MultiClassifier
from .classifiers import ClassifierUtil
from data.training_testing import Testing, Training
from util.Logger import Logger
import operator as op,collections as coll
from classification_attribute.feature_selection import feature_selection as feat_select
from data.util import unpickle_obj,flatten_train_set
from .results import ResSingle

__author__ = 'Kevin'

classification_logger=Logger(__name__)

def classify(classifier_util,settings,train_set,test_set,weights,print_res=True):
    """
    The main classification method

    :param settings:
    :param train_set:
    :param test_set:
    :param all_classifier_weights:
    :return:
    """

    classifier_to_accuracy=coll.defaultdict(lambda:[])
    """
    CLASSIFICATION WEIGHTS INITIALIZATION
    """
    start_weight,end_weight=weights.weights_range
    stepping=weights.stepping
    for classifiers_list in itertools.product(*[setting.classifier_list for setting in settings]):

        multi_classifier=MultiClassifier(classifiers_list,threshold=1,ll_ranking=False)
        classifier_names=str(multi_classifier)

        classification_logger.info("Classifying with {}".format(classifier_names))
        multi_classifier.fit(train_set.X,train_set.y)
        classification_logger.info("Fitting done, predicting with test set")



        use_prev=False

        for curr_weights in itertools.product(
            *itertools.repeat(np.arange(start_weight,end_weight+stepping,stepping),weights.num_classifiers)
        ):

            for i,s in enumerate(settings):
                s.weight=round(curr_weights[i],3)

            if all((i==0 for i in curr_weights)):
                continue

            classification_logger.debug("Using the weights {}".format(curr_weights))

            res_matrix=multi_classifier.predict_multi(test_set.X,classifier_weights=curr_weights,use_prev_prob=use_prev)
            use_prev=True # we cache the prediction, makes it faster to try out new weights

            if print_res:
                if (end_weight-start_weight)//stepping >3:
                    classification_logger.warn("Printing reject, too may possible weights")

                else:
                    classification_logger.info("Done, printing Results for {}".format(classifier_names))
                    classifier_util.print_res(settings,
                        y=test_set.y,
                        predictions=res_matrix,
                        ref_indexes=test_set.ref_index,
                        classifier_name=classifier_names,shorten_path=True)

            accuracy=classifier_util.calculate_accuracy(test_set.y,res_matrix)

            classifier_to_accuracy[classifier_names].append(
                ResSingle(accuracy=accuracy,weights=curr_weights,classifier_names=classifier_names))

    return classifier_to_accuracy

def feature_selection(settings,feature_selector_partial,train_sets,test_sets,num_attrs):
    train_Xs=[]
    test_Xs=[]

    for index,setting in enumerate(settings):
        train_set=train_sets[index]
        test_set=test_sets[index]

        #incase the num attr exceed the size
        total_num_attr=train_set.X.shape[1]
        setting.num_attribute=num_attrs[index]

        if total_num_attr<setting.num_attribute:
            setting.num_attribute=total_num_attr
            num_attrs[index]=setting.num_attribute

        num_genres=len(set(itertools.chain(*([i for i in i_list]for i_list in train_sets[index].y))))

        classification_logger.info("Currently doing feature selection on {}th data set with {}".format(index,str(feature_selector_partial)))
        classification_logger.info("Pre feature selection: num features: {}".format(train_set.X.shape[1]))

        feature_selector=feature_selector_partial(setting.num_attribute)
        train_X,test_X=feat_select(train_set,test_set,feature_selector,fit=True)


        classification_logger.info("Ending Dimension for train: {}".format(train_X.shape))
        classification_logger.info("Ending Dimension for test: {}".format(test_X.shape))

        train_Xs.append(train_X)
        test_Xs.append(test_X)

    return train_Xs,test_Xs

def load_training_testing(Xs,ys,ref_indexes,settings,train_set_index,test_set_index):
    """
    Load training and testing set based on indexes provided by crossvalidation

    :return: List of train_set and test_set objs
    """
    train_sets=[]
    test_sets=[]

    for c,setting in enumerate(settings):
        train_set=Training(setting,pickle_dir=setting.pickle_dir)
        train_set.X=Xs[c][train_set_index]
        train_set.y=ys[c][train_set_index]
        train_set.ref_index=ref_indexes[c][train_set_index]

        test_set=Testing(setting,pickle_dir=setting.pickle_dir)
        test_set.X=Xs[c][test_set_index]
        test_set.y=ys[c][test_set_index]
        test_set.ref_index=ref_indexes[c][test_set_index]

        train_sets.append(train_set)
        test_sets.append(test_set)

    #flatten training
    for train_set in train_sets:
        flatten_train_set(train_set)

    #make sure the sets match
    classification_logger.info("Checking the sets match")
    ys=[train_set.y for train_set in train_sets]
    ref_indexes=[train_set.ref_index for train_set in train_sets]

    test_ys=np.array([test_set.y for test_set in test_sets])
    test_ref_indexes=[test_set.ref_index for test_set in test_sets]

    for c,elem in enumerate((ys,ref_indexes,test_ys,test_ref_indexes)):

        prev=elem[0]
        match=True
        for e in elem[1:]:
            match=match and (e==prev).all()
        if not match:
            raise AttributeError("NOT MATCH FOR {} ELEMENT".format(c))

    return train_sets,test_sets


# def select_training_testing_sets(settings,Xs,y,ref_index,num,do_pickle=True):
#     """
#     Randomly choose from a super set of data and split it into a training set of size num. The remainder will become
#         the Test set. Uses _pick_random_samples
#
#     :param setting:
#     :param X:
#     :param y:
#     :param ref_index:
#     :param num:
#     :return: tuple_training,tuple_testing
#     """
#
#     selector=np.not_equal(ref_index,None)
#     ref_index=ref_index[selector]
#     Xs=[X[selector] for X in Xs]
#     y=y[selector]
#
#     train_Xs,train_y,train_ref_index,test_Xs,test_y,test_ref_index=_pick_random_samples(Xs,y,ref_index,num)
#
#     train_objs=[]
#     test_objs=[]
#     if do_pickle:
#         for c,setting in enumerate(settings):
#             train_X=train_Xs[c]
#             test_X=test_Xs[c]
#
#             _pickle_training_testing(setting,train_X,train_y,train_ref_index,test_X,test_y,test_ref_index)
#
#             training_obj=Training(label=setting,pickle_dir=setting.pickle_dir)
#             training_obj.set_data(train_X,train_y,train_ref_index)
#
#             testing_obj=Testing(label=setting,pickle_dir=setting.pickle_dir)
#             testing_obj.set_data(test_X,test_y,test_ref_index)
#
#             train_objs.append(training_obj)
#             test_objs.append(testing_obj)
#
#     return train_objs,test_objs