__author__ = 'Kevin'
import itertools

from analytics.classification_results.res_iterator import RightResultsIter,WrongResultsIter
from misc_scripts.assign_ref_index import global_ref_id

if __name__=="__main__":
    global_ref_id()

    #assign_ref_index_to_each_url()

    """
    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_chi_top4cls_10000"
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

"""
