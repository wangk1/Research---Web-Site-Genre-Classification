__author__ = 'Kevin'

from classification.classification_res import RightResultsIter,WrongResultsIter
import collections as coll,operator as op,itertools as it, os
from classification.util import pickle_obj,unpickle_obj
from analytics.graphics import plot_word_frequency,save_fig

def single_class_mispredition_freq(res_path):
    """
    Get the frequency of misprediction between single genre instances and the predicted genre

    :param res_path:
    :return:
    """

    print("Loading Iter")

    wrong_res_iter=WrongResultsIter.load_iter_from_file(res_path)
    right_res_iter=RightResultsIter.load_iter_from_file(res_path)

    genre_to_wrong_genre_count=coll.Counter()
    for c,res_obj in enumerate(it.chain(wrong_res_iter,right_res_iter)):
        if c%500==0:
            print(c)

        actual=res_obj.actual

        #single genre
        if len(actual)==1 and actual[0] != res_obj.predicted[0]:
            genre_to_wrong_genre_count.update([(actual[0],res_obj.predicted[0])])

    #plot
    plt=plot_word_frequency("Single Genre Mispredition",genre_to_wrong_genre_count)
    plt.tight_layout()
    save_fig("C:\\\\Users\\\\Kevin\\\\Desktop\\\\GitHub\\\\Research\\\\Webscraper\\\\classification_res\\\\genre_analysis\\\\single_miss.pdf",
             plt)

def top_level_cdf(res_folder):
    print("Loading Iter")

    wrong_res_iter=WrongResultsIter.load_iter_from_file(res_folder)
    right_res_iter=RightResultsIter.load_iter_from_file(res_folder)

    dist_count={}
    for c,res_obj in enumerate(it.chain(wrong_res_iter,right_res_iter)):
        dist_count[len(set(res_obj.actual))]=dist_count.get(len(set(res_obj.actual)),0)+1

    print(dist_count)

def multi_class_misprediction_freq(res_folder):
    """
    Look at multi class instances that are frequently mispredicted

    :param res_folder:
    :return:
    """


    print("Loading Iter")

    wrong_res_iter=WrongResultsIter.load_iter_from_file(res_folder)
    right_res_iter=RightResultsIter.load_iter_from_file(res_folder)

    genre_to_wrong_genre_count=coll.Counter()
    right_count=0
    for c,res_obj in enumerate(it.chain(wrong_res_iter,right_res_iter)):
        if c%500==0:
            print(c)

        actual=set(res_obj.actual)

        #check multiple genres
        if len(actual)>1:
            if not set(actual) <= set(res_obj.predicted[:len(actual)]):
                genre_to_wrong_genre_count.update([(tuple(actual),tuple(res_obj.predicted))])

                right_count+=1

    #sort the whole thing
    sorted_genre_to_wrong=sorted(genre_to_wrong_genre_count.items(),key=op.itemgetter(1),reverse=True)

    print(sorted_genre_to_wrong)
    print(right_count)


def load_prob_dict(path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\ll_prob_1"):
    ref_id_to_prob={}

    with open(path) as prob_path:
        for line in prob_path:
            components=line[:-1].split(",")
            ref_id_to_prob[components[0]]=dict((components[r+1],components[r]) for r in
                                                (r for r in range(0,len(components[1:]),2)))

    return ref_id_to_prob


def calculate_miss_classification_log_dist(res_folder,log_prob_dict):
    print("Loading Iter")

    wrong_res_iter=WrongResultsIter(res_folder)
    right_res_iter=RightResultsIter(res_folder)

    counts=coll.Counter()
    total_prob={}

    miss=0
    for res_obj in it.chain(wrong_res_iter,right_res_iter):
        actual=tuple(set(res_obj.actual))

        if actual[0] != res_obj.predicted[0]:

            diff=abs(abs(float(log_prob_dict[str(res_obj.ref_id)][actual[0]]))-abs(float(log_prob_dict[str(res_obj.ref_id)][res_obj.predicted[0]])))

            counts.update([(actual[0],res_obj.predicted[0])])
            total_prob[(actual[0],res_obj.predicted[0])]=total_prob.get((actual[0],res_obj.predicted[0]),0)+diff

            with open("ll_wrong",mode="a") as wrong_w:
                wrong_w.write("{},{}\n".format(res_obj.ref_id,','.join(sorted(log_prob_dict[str(res_obj.ref_id)].values()[1:]))))


        #if not set(actual) <= set(res_obj.predicted[:len(actual)]):
            #counts.update([(actual,res_obj.predicted[:len(actual)])])
            #total_prob[(actual,res_obj.predicted[:len(actual)])]+=1

    for k in total_prob.keys():
        total_prob[k]/=counts[k]

    #now print it all out
    sorted_list=[i for i in sorted(total_prob.items(),key=op.itemgetter(1),reverse=False) if i[1]<3]

    print(sorted_list)

def consensus_class(res_path):

    ref_id_to_pred_and_actual={}

    print("Loading Iter")

    wrong_res_iter=WrongResultsIter.load_iter_from_file(res_path)
    right_res_iter=RightResultsIter.load_iter_from_file(res_path)

    #just grab the mispredictions
    for res_obj in it.chain(wrong_res_iter,right_res_iter):
        ref_id=res_obj.ref_id
        actual=tuple(set(res_obj.actual))
        pred=tuple(sorted(res_obj.predicted[:len(actual)]))

        #get some wrong sample
        if sum(i not in pred for i in actual)==0:
            continue

        pred_actual_dict=ref_id_to_pred_and_actual.get(ref_id,{})
        pred_actual_dict["actual"]=actual
        #add new entries

        pred_counter=pred_actual_dict.get("pred",coll.Counter())
        pred_counter.update([pred])

        pred_actual_dict["pred"]= pred_counter

        ref_id_to_pred_and_actual[ref_id]=pred_actual_dict

    actual_to_majority_vote=coll.Counter()
    #now iter over k,v pairs
    for k,v in ref_id_to_pred_and_actual.items():
        actual_tuple=v["actual"]
        pred_counter=v["pred"]

        #skip if no consensus
        if all((v<3 for k,v in pred_counter.items())):
            continue

        pred_tuple=tuple(sorted(pred_counter.items(),key=op.itemgetter(1),reverse=True)[0][0])

        actual_to_majority_vote.update([(actual_tuple,pred_tuple)])

    print(sorted(actual_to_majority_vote.items(),key=op.itemgetter(1),reverse=True))

def count_num_multi_predict(res_path):

    ref_id_set=set()

    wrong_res_iter=WrongResultsIter.load_iter_from_file(res_path)
    right_res_iter=RightResultsIter.load_iter_from_file(res_path)

    count=0
    single_genre=0
    for res_obj in it.chain(wrong_res_iter,right_res_iter):
        if res_obj.ref_id not in ref_id_set and len(set(res_obj.actual)) >1:
            count+=1
            ref_id_set.add(res_obj.ref_id)
        elif res_obj.ref_id not in ref_id_set and len(set(res_obj.actual)) ==1:
            single_genre+=1
            ref_id_set.add(res_obj.ref_id)

    print(count)
    print(single_genre)

def frequently_predicted_class(res_path,top_x=2):
    """
    Top x frequently predicted together class. The tuple of genre and genre is sorted so there is no repeats.

    :param res_path:
    :param top_x:
    :return:
    """

    print("Loading Iter")

    wrong_res_iter=WrongResultsIter.load_iter_from_file(res_path)
    right_res_iter=RightResultsIter.load_iter_from_file(res_path)

    predicted_counter=coll.Counter()
    actual_counter=coll.Counter()


    predicted_counter.update((tuple(sorted(p)) for p in (
        res_obj.predicted[:top_x] for res_obj in it.chain(wrong_res_iter,right_res_iter) if len(res_obj.actual)>1)

                              ))
    actual_counter.update((tuple(sorted(p)) for p in (
        tuple(set(res_obj.actual)) for res_obj in it.chain(wrong_res_iter,right_res_iter) if len(set(res_obj.actual))>1)

                           ))

    print("Predicted")
    print(predicted_counter)
    print("Actual")
    print(actual_counter)
