import itertools

__author__ = 'Kevin'
import os,re,collections,operator
from util.Logger import Logger
from analytics.graphics import plot_word_frequency,save
from classification.classification_res import WrongResultsIter, ClassificationResultInstance

WrongPrediction=collections.namedtuple("WrongPrediction",("ref_id","actual","predicted","classifier"))
def get_wrongly_predicted_samples(filepath):
    """
    Convert the wrong prediction file to list of object, WrongPrediction object, format.

    :return:
    """
    wrong_prediction=[]

    classifier=re.search(".*(?=_wrong.txt)",filepath.split("\\")[-1]).group(0)
    with open(filepath) as wrong_sample_file:
        for line in wrong_sample_file:

            res_list=line.split(" ")

            wrong_prediction.append(WrongPrediction(ref_id=res_list[0][:-1],predicted=res_list[2][:-1],actual=res_list[4][:-1],
                                                    classifier=classifier))

    return wrong_prediction

def count_miss_ratio():
    """
    Read from the result files and count the number of times each missed testing sample is missed out of the total classifiers

    :return:
    """
    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\chi2_1k"
    
    wrong_file_list=[os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and re.match("^result.*wrong[.]txt$",f)]

    count=collections.Counter()

    for f in wrong_file_list:
        print(f)
        count.update(map(lambda o: o.ref_id,get_wrongly_predicted_samples(f)))
    print(len(count))

    print(sorted(count.items(),key=operator.itemgetter(1),reverse=True))

def transform_to_each_genre_missed(missed_samples,by_classifer=False):
    """
    Count the number of times each genre was missed classified as another genre and return a map of the mapping

    :param missed_samples: List of WrongPrediction objects
    :return: Dictionary of acutal genres to a counter of predicted genre counts
    """

    if not by_classifer:
        actual_to_predicted=collections.defaultdict(lambda: collections.Counter()) #maps actual genre to predicted and how many times they are missed
    else:
        #maps classifier -> actual_genre -> predicted_genre and count
        actual_to_predicted=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.Counter()))

    for missed_sample_obj in missed_samples:
        actual_genre=missed_sample_obj.actual
        predicted_genre=missed_sample_obj.predicted
        classifier=missed_sample_obj.classifier

        if by_classifer:
            actual_to_predicted[classifier][actual_genre].update([predicted_genre])

        else:
            actual_to_predicted[actual_genre].update([predicted_genre])

    return actual_to_predicted


def plot_miss_per_genre(path,outpath,classifiers=None):
    """
    Given the path to classification result folder of multiple classifier.

    produce a plot of the classifier's misses.

    :param path: the input folder where the classifiers' result(s) are
    :param classifiers: A set of classifier whose results to graph. Note that if none, all of classifier's
        results will be combined.

    :return:
    """

    #grab the actual misses, counter in default dict in default dict. First layer for classifiers, second layer
    #is for correct genres, finally the counter is to count how many times it got miss classified as somethine else
    classifier_to_misses_genre=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.Counter()))
    for true_miss in (w for w in WrongResultsIter(path,classifiers) if not w.is_swing_sample()):
        assert isinstance(true_miss,ClassificationResultInstance)

        classifier_to_misses_genre[true_miss.classifier][true_miss.actual].update([true_miss.predicted])


    #now plot each one, output to OUTPUT/classifier
    for classifier, actual_to_miss in classifier_to_misses_genre.items():
        for actual_genre,miss_freq in actual_to_miss.items():
            plt=plot_word_frequency("{}-{} Misclassifications".format(classifier,actual_genre),miss_freq,plot_top=len(miss_freq))

            out_path=os.path.join(outpath,classifier)
            if not os.path.exists(out_path):
                os.mkdir(out_path)

            save("{}/{}_miss_true.pdf".format(out_path,actual_genre),plt)
            plt.close()



