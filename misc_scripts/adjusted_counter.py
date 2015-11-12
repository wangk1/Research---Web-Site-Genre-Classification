import collections
import os
import re


from classification.classification_res import get_classification_res
from db.db_model.mongo_websites_models import URLBow
from util.base_util import normalize_genre_string

__author__ = 'Kevin'




def calculate_adjusted_miss_rate(res_folder):
    """
    Given the path to folder containing classifier results files that have the form of ClassifierName_wrong.txt or *_right.txt.

    We adjust for those wrong examples that have 1+ class and

    :param path_of_classifier_res:
    :return:
    """
    """
    Script that reads how many wrong we have
    """

    #counter the number of instances classified as wrong, but is actually predicted
    #into one of its many genres
    swing_counter=collections.Counter()
    total_wrong_size=0
    total_right_size=0
    num_wrong=0
    num_right=0

    #get all result files that ends with _right or _wrong
    for a_result in filter(lambda x: os.path.isfile(os.path.join(res_folder,x)),os.listdir(res_folder)):
        assert isinstance(a_result,str)
        abs_result=os.path.join(res_folder,a_result)
        print(a_result)

        right=0
        wrong=0
        if a_result.find("right") > -1:
            #count the rights
            with open(abs_result) as file:
                right+=sum((1 for i in file if i.strip() != ""))

        elif a_result.find("wrong")>-1:
            wrong_res_objs=get_classification_res(abs_result)

            #grab all the genres and see if it exists
            for c,res_obj in enumerate(wrong_res_objs):
                found=False

                #grab all short genres and see if it matches
                url_bow_obj=URLBow.objects(index=res_obj.ref_id).only("short_genres")[0]

                found=res_obj.predicted in (normalize_genre_string(g,1) for g in url_bow_obj.short_genres) or found

                if found:
                    swing_counter.update([res_obj.ref_id])
                    right+=1

                else:
                    wrong+=1

        print("Total right: {}, total wrong: {}".format(right,wrong))

    print("Swing counter {}".format(str(swing_counter)))
    print("Swing counter size : {}".format(len(swing_counter)))
    print("Average bow size of rights: {}. For wrong: {}".format(total_right_size/num_right,total_wrong_size/num_wrong))

