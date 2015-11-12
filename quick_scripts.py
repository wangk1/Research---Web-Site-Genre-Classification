__author__ = 'Kevin'
from misc_scripts.adjusted_counter import calculate_adjusted_miss_rate
from classification.classification_res import RightResultsIter


if __name__=="__main__":
    #calculate_adjusted_miss_rate("C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_2000_chi2")
    print(len([i for i in RightResultsIter("C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_10000_chi2")]))
