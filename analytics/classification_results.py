import itertools

__author__ = 'Kevin'
import os,re,collections,operator
from util.Logger import Logger
from analytics.graphics import plot_word_frequency,save

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


def plot_miss_per_genre(path,outpath,by_classifier=False):
    """
    Given the path to

    :param path:
    :return:
    """

    wrong_file_list=[os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and re.match("^.*_wrong[.]txt$",f)]

    Logger.info("found the following files: {}".format(str(wrong_file_list)))

    count=collections.Counter()

    actual_to_missed_genre_mapping=transform_to_each_genre_missed(
                itertools.chain(*(get_wrongly_predicted_samples(f) for f in wrong_file_list)),by_classifier)

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    #now plot each one
    for actual_genre, inner_dict in actual_to_missed_genre_mapping.items():
        #total_sum=sum((i[1] for i in counter.items()))
        #counter=dict(((k,v/total_sum) for k,v in counter.items()))

        if not by_classifier:
            plt=plot_word_frequency("{} Misclassifications".format(actual_genre),inner_dict,plot_top=len(inner_dict))
            save("{}/{}_miss.pdf".format(outpath,actual_genre),plt)
        else:
            for classifier in actual_to_missed_genre_mapping.keys():
                Logger.info("Plotting for classifier {}".format(classifier))
                for actual_genre in actual_to_missed_genre_mapping[classifier].keys():
                    inner_dict=actual_to_missed_genre_mapping[classifier][actual_genre]
                    out_path=os.path.join(outpath,classifier)

                    os.path.exists(out_path) or os.makedirs(out_path)

                    plt=plot_word_frequency("{} Misclassifications".format(actual_genre),inner_dict,plot_top=len(inner_dict))
                    save("{}/{}_miss.pdf".format(out_path,actual_genre),plt)
                    plt.close()



