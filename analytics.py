import functools

__author__ = 'Kevin'

from analytics.combine_genre_analysis import *
from analytics.mutual_information import *
from analytics.genre_similarity import get_genre_similarities
from db.db_model.mongo_websites_models import TrainSetBow

from analytics.dmoz_alexa_similarity import dmoz_alexa_similarity
from analytics.genre_analytics.genre_count import num_genre_per_webpage
from data.util import unpickle_obj
from analytics.genre_analytics.genre_count import tabulate_genre_dist

def calculate_top_percent():
    import db.db_collections.mongo_collections as coll, operator

    #load bow from all 30 classes
    bows=dict((i.short_genre,i.bow) for i in coll.Top30WordGenre().iterable())

    most_common={} #dictionary of the most common top 30 words

    for short_genre,bow in bows.items():
        with open("top_30_stats.txt",mode="a",encoding="latin_1",errors="ignore") as file:
            file.write("Generating statistics for short genre {}".format(short_genre)+"\n")

            #find how many in common
            for w,_ in bow.items():
                occurences=functools.reduce(lambda count,mi_obj: count+(1 if w in operator.itemgetter(1)(mi_obj) else 0),bows.items(),0)

                file.write("{} occured {} times".format(w,occurences)+"\n")

                #keep track of the count
                if w not in most_common:
                    most_common[w]=occurences


    #most common top 30 word
    with open('top_30.txt',mode='a',encoding="latin_1",errors="ignore") as file:
        sorted_list=sorted(most_common.items(),key=operator.itemgetter(1),reverse=True)

        for (w,c) in sorted_list:
            file.write("{}, {}\n".format(w,c))




if __name__=="__main__":
    dmoz_alexa_similarity()
    exit(0)
    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_chi_top1cls_10000"
    outpath="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\summary_2000_chi2\\miss_plt"

    y_path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir\\y_summary_pickle"

    y=unpickle_obj(y_path)

    tabulate_genre_dist(y)

    #num_genre_per_webpage("C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\pickle_dir\\y_summary_pickle")


    #dmoz_alexa_similarity()
    # #prob_dict=load_prob_dict()
    #for i in range(1,5):
        #consensus_count,consensus_total=consensus_class_per_genre(path,filter_func=lambda x:len(x)==i)
        #plot_consensus_percentile(consensus_count,consensus_total)
    #multi_class_misprediction_freq(path)

    #plot_miss_per_genre(path,outpath,classifiers="LogisticRegression")


    #mutual_information_similarity("genre_similarity.txt")







