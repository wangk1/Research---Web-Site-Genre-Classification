__author__ = 'Kevin'

import collections as coll
import operator as op
import itertools as it

from matplotlib import pyplot

import analytics.graphics as graphics
from analytics.classification_results.res_iterator import RightResultsIter,WrongResultsIter
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

def consensus_class(res_path,top_prediction=2,filter_func=lambda x:len(x)==4):
    """
    Calculates the portions of the predictions that agrees with the classes's multiple classes

    For example: if


    :param res_path:
    :return:
    """
    #dictionary to hold the xth class and the number of agreements for it
    consensus_count=coll.defaultdict(lambda:0)
    consensus_total=coll.defaultdict(lambda:0)

    ref_id_to_pred_and_actual={}

    print("Loading Iter")

    #wrong_res_iter=WrongResultsIter.load_iter_from_file(res_path)
    right_res_iter=RightResultsIter.load_iter_from_file(res_path)

    #gather together all of an instance's data and id
    right_res_instances=coll.defaultdict(lambda: [])

    for res_obj in right_res_iter:
        right_res_instances[res_obj.ref_id].append(res_obj)


    #just grab the mispredictions
    for ref_id,list_of_res_obj in right_res_instances.items():
        actual=tuple(set(list_of_res_obj[0].actual))

        #Todo:change here
        if not filter_func(actual):
            continue

        genre_hit_count=[0]*len(actual)

        genre_by_consensus=sorted([sum(g in set(res_obj.predicted[:top_prediction]) for res_obj in list_of_res_obj) for g in actual]
               ,reverse=True)


        for c,g_hit in enumerate(genre_by_consensus):
            genre_hit_count[c]+=g_hit

        #now we sort the the hit for each class
        #genre_hit_count=sorted(genre_hit_count,reverse=True)

        for class_num in range(0,len(actual)):
            consensus_count[class_num]+=genre_hit_count[class_num]
            consensus_total[class_num]+=6

    print(sorted(consensus_count.items()))
    print(sorted(consensus_total.items()))


def consensus_class_per_genre(res_path,top_prediction=2,filter_func=lambda x:len(x)==4):
    """
    Get the consensus for each genre, return the list

    :param res_path:
    :param top_prediction:
    :param filter_func:
    :return:
    """
    #dictionary to hold the xth class and the number of agreements for it
    consensus_count=coll.defaultdict(lambda:coll.defaultdict(lambda:[]))
    consensus_total=coll.defaultdict(lambda:coll.defaultdict(lambda:[]))

    ref_id_to_pred_and_actual={}
    num_classes=0

    print("Loading Iter")

    #wrong_res_iter=WrongResultsIter.load_iter_from_file(res_path)
    right_res_iter=RightResultsIter.load_iter_from_file(res_path)

    #gather together all of an instance's data and id
    right_res_instances=coll.defaultdict(lambda: [])

    for res_obj in right_res_iter:
        right_res_instances[res_obj.ref_id].append(res_obj)


    #just grab the mispredictions
    for ref_id,list_of_res_obj in right_res_instances.items():
        actual=tuple(set(list_of_res_obj[0].actual))

        #Todo:change here
        if not filter_func(actual):
            continue

        genre_consensus={g:sum(g in set(res_obj.predicted[:top_prediction]) for res_obj in list_of_res_obj) for g in actual}
        num_classes=len(genre_consensus)

        for index,(g,count) in enumerate(sorted(genre_consensus.items(),key=op.itemgetter(1),reverse=True)):
            consensus_count[index][g].append(count)
            consensus_total[index][g].append(6)

        #now we sort the the hit for each class
        #genre_hit_count=sorted(genre_hit_count,reverse=True)



    return consensus_count,consensus_total

def plot_total_consensus(consensus_count,consensus_total):
    consensus_count=sorted(consensus_count.items(),key=lambda entry:entry)
    num_classes=len(consensus_count)

    pyplot.close()
    pyplot.figure(1)

    for c in range(0,num_classes):

        ax=pyplot.subplot(num_classes,1,c)
        genre_dict=consensus_count[c][1]
        genre_total_dict=consensus_total[c]

        genre_to_counts=[]

        for genre,count in genre_dict.items():
            genre_to_counts.append((genre,sum(count),sum(genre_total_dict[genre])))

        genre_to_counts=sorted(genre_to_counts,key=lambda t:t[0])

        pyplot.hold(True)
        pyplot.title("Consensus plot for Genre {}".format(c))


        pyplot.bar(range(len(genre_to_counts)),[g[2] for g in genre_to_counts],color='#deb0b0',label="Consensus Total",align='center')
        pyplot.bar(range(len(genre_to_counts)),[g[1] for g in genre_to_counts],color='#b0c4de',label="Consensus Counts",align='center')

        pyplot.xticks(range(len(genre_to_counts)),[g[0] for g in genre_to_counts],size= 5)
        legend=pyplot.legend(loc="upper right")
        legend.set_visible(False)
        pyplot.hold(False)


    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\consensus_plots\\total_{}.pdf"
    graphics.save_fig(path.format(num_classes),pyplot)
    pyplot.close()

    print("Done")

    print("Done")

def plot_consensus_percentile(consensus_count,consensus_total):
    """
    Uses the 90th percentile plot.

    :param consensus_count:
    :param consensus_total:
    :return:
    """

    consensus_count=sorted(consensus_count.items(),key=lambda entry:entry)
    num_classes=len(consensus_count)

    pyplot.figure(1)
    for c in range(0,num_classes):

        ax=pyplot.subplot(num_classes,1,c)
        genre_dict=consensus_count[c][1]
        genre_total_dict=consensus_total[c]

        genre_to_counts=[]

        for genre,count in genre_dict.items():
            genre_to_counts.append((genre,count,genre_total_dict[genre]))

        genre_to_counts=sorted(genre_to_counts,key=lambda t:t[0])

        pyplot.title("Consensus plot for Genre {}, total number of instances {}".format(c,sum(it.chain(*(g[2] for g in genre_to_counts)))/6))

        #set up xaxis labels
        pyplot.xticks(list(range(1,len(genre_to_counts)+1)),[g[0] for g in genre_to_counts])
        pyplot.tick_params(axis='both', which='major', labelsize=5)

        #now plot y axis
        for index,res in enumerate(genre_to_counts):
            graphics.add_bar_plot(index+1,res[1])

        #pyplot.xticks(range(len(genre_to_counts)),["0"]+[g[0] for g in genre_to_counts],size= 5)
        pyplot.legend(loc="upper right")


    pyplot.tight_layout()

    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\consensus_plots\\percentile_{}.pdf"
    graphics.save_fig(path.format(num_classes),pyplot)
    pyplot.close()

    print("Done")


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
