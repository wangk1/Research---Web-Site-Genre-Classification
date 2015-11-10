import collections
import os
import re


from classification.classification_res import get_classification_res
from db.db_model.mongo_websites_models import URLBow
from util.base_util import normalize_genre_string

__author__ = 'Kevin'




def calculate_adjusted_miss_rate(path_of_classifier_res):
    """
    Given the path to folder containing classifier results files that have the form of ClassifierName_wrong.txt or *_right.txt.

    We adjust for those wrong examples that have 1+ class and

    :param path_of_classifier_res:
    :return:
    """
    """
    Script that reads how many wrong we have
    """

    res_folder="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res"
    folder_patterns="^summary_.*_chi2$"

    all_folders=[folder for folder in (os.path.join(res_folder,f) for f in os.listdir(res_folder)
                                       if re.match(folder_patterns,f)
                                       )]
    all_folders=sorted(all_folders)
    for path in all_folders:
        print(path)


        #get all result files that ends with _right or _wrong
        for a_result in filter(lambda x: os.path.isfile(os.path.join(path,x)),os.listdir(path)):
            assert isinstance(a_result,str)
            abs_result=os.path.join(path,a_result)
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
                        right+=1

                    else:
                        wrong+=1



            print("Total right: {}, total wrong: {}".format(right,wrong))
    print(all_folders)