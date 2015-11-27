import collections
import os
import re,operator


from classification.classification_res import get_classification_res, RightResultsIter,WrongResultsIter,ClassificationResultInstance
from db.db_model.mongo_websites_models import URLBow
from util.base_util import normalize_genre_string
from util.Logger import Logger


__author__ = 'Kevin'


def get_result_distribution(path,classifier):
    counts=[0]*3

    for res in RightResultsIter(path,classifier):
        assert isinstance(res,ClassificationResultInstance)

        counts[res.predicted.index(res.__actual)]+=1

    print(counts)


def calculate_average_page_size(res_folder):
    total_bow_sizes={"right":0,"wrong":0,"swing":0}
    bow_count={"right":0,"wrong":0,"swing":0}


    Logger.info("Average bow size, on right bow size")
    for right_res in RightResultsIter(res_folder):
        total_bow_sizes["right"]+=len(URLBow.objects.get(index=right_res.ref_id).bow)
        bow_count["right"]+=1

    Logger.info("Average bow size, on wrong bow size")
    for wrong_res in WrongResultsIter(res_folder):
        if wrong_res.is_swing_sample():
            label="swing"
        else:
            label="wrong"

        bow_count[label]+=1
        total_bow_sizes[label]+=len(URLBow.objects.get(index=wrong_res.ref_id).bow)

    print([(label,total/bow_count[label],bow_count[label]) for label,total in total_bow_sizes.items()])

def calculate_average_bow_size(res_folder):
    """
    Calculate average bow size for the URLBow database

    :param res_folder:
    :return:
    """

    total_bow_sizes={"right":0,"wrong":0,"swing":0}
    bow_count={"right":0,"wrong":0,"swing":0}

    Logger.info("Average bow size, on right bow size")
    for right_res in RightResultsIter(res_folder):
        total_bow_sizes["right"]+=len(URLBow.objects.get(index=right_res.ref_id).bow)
        bow_count["right"]+=1

    Logger.info("Average bow size, on wrong bow size")
    for wrong_res in WrongResultsIter(res_folder):
        if wrong_res.is_swing_sample():
            label="swing"
        else:
            label="wrong"

        bow_count[label]+=1
        total_bow_sizes[label]+=len(URLBow.objects.get(index=wrong_res.ref_id).bow)

    print([(label,total/bow_count[label] if bow_count[label] != 0 else 1,bow_count[label]) for label,total in total_bow_sizes.items()])


def calculate_genres_per_instance(res_folder,classifiers=""):
    current_classifier=classifiers

    right_genresize_counter=collections.Counter()
    wrong_genresize_counter=collections.Counter()
    swing_genresize_counter=collections.Counter()

    Logger.info("Current on rights")


    #iterate over the right samples first, we don't write to file because right files are the same
    for right_res_obj in {x.ref_id: x for x in RightResultsIter(res_folder,classifiers)}.values():
        assert isinstance(right_res_obj,ClassificationResultInstance)
        if right_res_obj.classifier != current_classifier:
            current_classifier=right_res_obj.classifier

        #now find the size of its genre
        right_genresize_counter.update([len(URLBow.objects.get(index=right_res_obj.ref_id).short_genres)])

    Logger.info("Current on wrongs")

    swing_file=res_folder+"/{}swing.txt".format(classifiers+"_" if classifiers.strip()!="" else classifiers)
    wrong_file=res_folder+"/{}wrong_true.txt".format(classifiers+"_" if classifiers.strip()!="" else classifiers)

    with open(swing_file,mode="w") as swing_handle,open(wrong_file,mode="w") as wrong_handle:
        #iterate over the wrong samples
        for wrong_res_obj in {x.ref_id: x for x in WrongResultsIter(res_folder,classifiers)}.values():
            assert isinstance(wrong_res_obj,ClassificationResultInstance)
            if wrong_res_obj.classifier != current_classifier:
                current_classifier=wrong_res_obj.classifier

            if wrong_res_obj.is_swing_sample():
                swing_handle.write(str(wrong_res_obj)+"\n")

                swing_genresize_counter.update([len(URLBow.objects.get(index=wrong_res_obj.ref_id).short_genres)])

            else:
                wrong_handle.write(str(wrong_res_obj)+"\n")

                #now find the size of its genre
                wrong_genresize_counter.update([len(URLBow.objects.get(index=wrong_res_obj.ref_id).short_genres)])

    print("Wrong predicted sample distrbution: {}".format(sorted(wrong_genresize_counter.items(),key=operator.itemgetter(0))))
    print("Right predicted sample distrbution: {}".format(sorted(right_genresize_counter.items(),key=operator.itemgetter(0))))
    print("Swing sample distrbution: {}".format(sorted(swing_genresize_counter.items(),key=operator.itemgetter(0))))


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

