__author__ = 'Kevin'
from misc_scripts.adjusted_counter import calculate_adjusted_miss_rate,calculate_genres_per_instance,calculate_average_bow_size
from classification.classification_res import RightResultsIter
from misc_scripts.adjusted_counter import get_result_distribution


if __name__=="__main__":
    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_chi_top1cls_1000"
    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_chi_top1cls_{}"
    classifier="LogisticRegression"

    #get_result_distribution(path,classifier)
    for c in {500,1000,5000,8000,10000,20000,30000,15000}:
        print("Working on {}".format(path.format(c)))
        calculate_genres_per_instance(path.format(c),classifier)

    #calculate_average_bow_size(path)

    #calculate_adjusted_miss_rate("C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_10000_chi2")
