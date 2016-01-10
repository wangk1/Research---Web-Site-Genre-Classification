import itertools
from classification.classifiers import MultiClassifier
from .classifiers import Classifier
from util.Logger import Logger
import operator as op

__author__ = 'Kevin'

classification_logger=Logger(__name__)

def classify(settings,train_set,test_set,all_classifier_weights,print_res=True):
    """
    The main classification method

    :param settings:
    :param train_set:
    :param test_set:
    :param all_classifier_weights:
    :return:
    """

    classifier_to_accuracy={}
    for classifiers_list in itertools.product(*[setting.classifier_list for setting in settings]):

        multi_classifier=MultiClassifier(classifiers_list,threshold=1,ll_ranking=False)
        classifier_names=str(multi_classifier)

        classification_logger.info("Classifying with {}".format(classifier_names))
        multi_classifier.fit(train_set.X,train_set.y)
        classification_logger.info("Fitting done, predicting with test set")

        #CLASSIFICATION, adjust weights
        classifier_util=Classifier()

        use_prev=False

        for curr_weights in next(all_classifier_weights):
            if all((i==0 for i in curr_weights)):
                continue

            classification_logger.info("Using the weights {}".format(curr_weights))

            res_matrix=multi_classifier.predict_multi(test_set.X,classifier_weights=curr_weights,use_prev_prob=use_prev)
            use_prev=True # we cache the prediction, makes it faster to try out new weights

            if print_res:
                print("Done, printing Results for {}".format(classifier_names))
                classifier_util.print_res(settings,
                    y=test_set.y,
                    predictions=res_matrix,
                    ref_indexes=test_set.ref_index,
                    classifier_name=classifier_names)

            accuracy=classifier_util.calculate_accuracy(test_set.y,res_matrix)

            #best the best classifier and weights associated with it
            classifier_to_accuracy[classifier_names]=max(classifier_to_accuracy.get(classifier_names,(0,("w1","w2"))),
                                                        tuple([accuracy])+tuple([curr_weights]))
            print("Best so far {}".format(max(classifier_to_accuracy.items(),key=op.itemgetter(1))))

    return classifier_to_accuracy
