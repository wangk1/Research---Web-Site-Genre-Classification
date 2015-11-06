import functools

__author__ = 'Kevin'

from analytics.mutual_information import *
from analytics.genre_similarity import get_genre_similarities
from db.db_model.mongo_websites_models import TrainSetBow

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
    calculate_training_set_mu(TrainSetBow)

    #mutual_information_similarity("genre_similarity.txt")

    """
    with open("C:/Users/Kevin/Desktop/GitHub/Research/Webscraper/top_30_stats.txt") as file:
        #read each line from file
        dict_occurence={}
        prev_title=""
        for line in file:
            if line.startswith("Generating statistics for short genre"):
                #create new file
                if not prev_title=="":
                    plt=graphics.plot_word_frequency(prev_title,dict_occurence)
                    graphics.save("freq_count/{}.pdf".format(prev_title.replace("/","_").replace("\n","")),plt)
                prev_title=line.split(" ")[-1]

                dict_occurence={}

            else:
                words=line.split(" ")

                dict_occurence[words[0]]=int(words[2])

        #dump last one
        plt=graphics.plot_word_frequency(prev_title,dict_occurence)
        graphics.save("freq_count/{}.pdf".format(prev_title.replace("/","_").replace("\n","")),plt)
    """





