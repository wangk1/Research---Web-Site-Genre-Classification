__author__ = 'Kevin'
from misc_scripts.adjusted_counter import calculate_adjusted_miss_rate,calculate_genres_per_instance,calculate_average_bow_size
from classification.classification_res import RightResultsIter,WrongResultsIter
from misc_scripts.adjusted_counter import get_result_distribution
from misc_scripts.grab_urls_and_genres import grab_urls_and_genres
import itertools

if __name__=="__main__":
    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_chi_top1cls_500"
    #path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_100_chi_truncated_lsa"
    classifier="LogisticRegression"
    num_top=1

    correct=0
    wrong=0

    for res_instance in itertools.chain(*[RightResultsIter(path,classifier),WrongResultsIter(path,classifier)]):
        if set(res_instance.actual) <= set(res_instance.predicted[:num_top]) :
            correct+=1
        else:
            wrong+=1

    print("{},{}".format(correct,wrong))


