import itertools
from classification.classifiers import MultiClassifier
from .classifiers import Classifier
from util.Logger import Logger

__author__ = 'Kevin'

classification_logger=Logger(__name__)

def classify(settings,train_set,test_set,classifier_weights,print_res=True):
    """
    The main classification method

    :param settings:
    :param train_set:
    :param test_set:
    :param classifier_weights:
    :return:
    """

    classifier_to_accuracy={}
    for classifiers_list in itertools.product(*[setting.classifier_list for setting in settings]):

        multi_classifier=MultiClassifier(classifiers_list,threshold=1,ll_ranking=False)
        classifier_name=str(multi_classifier)

        #CLASSIFICATION
        classifier_util=Classifier()
        res_matrix=_classify_single(train_set,test_set,multi_classifier,classifier_weights)


        print("Done, printing Results for {}".format(classifier_name))
        print_res and classifier_util.print_res(settings,
            y=test_set.y,
            predictions=res_matrix,
            ref_indexes=test_set.ref_index,
            classifier_name=classifier_name)

        accuracy=classifier_util.calculate_accuracy(test_set.y,res_matrix)
        print("Accuracy is {} for {}".format(accuracy,classifier_name))
        classifier_to_accuracy[classifier_name]=accuracy
    return classifier_to_accuracy

def _classify_single(train_set,test_set,classifier,classifier_weights,increment=500):
        """
        Classify with test_X and generate result files for each classifier set

        :param classifiers: set of additional classifier to test with in addition to self.classifier classifiers
        :return:
        """

        classifier_name=str(classifier)

        classification_logger.info("Classifying with {}".format(classifier_name))
        classifier.fit(train_set.X,train_set.y)

        classification_logger.info("Fitting done, predicting with test set")

        res=classifier\
                .predict_multi(test_set.X,classifier_weights=classifier_weights)

        return res