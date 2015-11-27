import operator,itertools
import math
from db.db_model.mongo_websites_models import TrainSetBow,WordCount_training,GenreCount_training,TopWordGenre\
                                    , MutualInformationGenres
import analytics.graphics as graphics
import collections
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from db.db_collections.mongo_collections import MutualInformation
from pymongo import database as db
from mongoengine import connection
from db.settings import MUTUAL_INFO_DB
import operator,itertools


__author__ = 'Kevin'



def get_all_mi_and_plot(reversed=False):
    """
    Grab all mutual information data from the database collection MutualInformation and plot them with matlibplot

    :return: None!
    """
    #graphics.plot_save_all_genre()
    mi=MutualInformation()


    for mi_obj in mi.iterable():

        genre=mi_obj["short_genre"]
        bow_mi=mi_obj["bow"]

        filtered_bow_mit={}
        for k,v in bow_mi.items():
            if not k.isdigit():
                filtered_bow_mit[k]=v

        plt=graphics.plot_word_frequency(genre,filtered_bow_mit,reversed=reversed)
        graphics.save_fig("graphs/{}.pdf".format(("reversed_" if reversed else "")+genre.replace("/","_")),plt)

        print(genre)

def mutual_information_similarity(file_name):
    """
    Calculates MI between all pairs of short_genre based on their word's MI.

    Prints to file the similarity

    :return:
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    import math

    SimilarityScore=collections.namedtuple("SimilarityScore",("g1","g2","score")) #a type

    #fetch all short genres
    mi_coll=MutualInformation()
    #all possible pairs of genre with no repeat
    genres=[]

    #calculate cosine similarity b/w pairs
    dv=DictVectorizer()

    def extract_bow_add_to_genres(genre,bow):
        if genre not in genres:
            genres.append(genre)

        new_bow={}

        for k in bow.keys():

            curr=bow[k]
            new_bow[k]=0 if math.isnan(curr) or math.isinf(curr) else curr

            new_bow==0 and print("Eliminated element")

        return new_bow


    bow_matrix=dv.fit_transform(extract_bow_add_to_genres(mi_obj.short_genre,mi_obj.bow) for mi_obj in mi_coll.iterable())

    print("Done with making vector")
    #sort the pairs by the cosine similarity score
    similarity_matrix=cos_sim(bow_matrix)

    print("Done with similarity calculation")
    sorted_list=[]
    #sort the similarity scores
    for x,y in itertools.combinations(range(0,len(genres)),2):
        sorted_list.append(SimilarityScore(genres[x],genres[y],similarity_matrix[x][y]))
    #sort!
    sorted_list=sorted(sorted_list,key=operator.itemgetter(2),reverse=True)

    print("printing file")
    with open(file_name,mode="a",errors="ignore",encoding="latin-1") as file:
        for l in sorted_list:
            file.write("{}, {} value: {}\n".format(l[0],l[1],l[2]))


def _calc_word_genre_counts(training_coll_cls):
    assert training_coll_cls==TrainSetBow

    db=connection.get_db(MUTUAL_INFO_DB)

    genre_word_update_template="bow.{}"
    for count,training_obj in enumerate(training_coll_cls.objects):
        if count % 1000 ==0:
            print("count is at {}".format(count))

        single_bow=training_obj.bow

        word_count_update={}
        #update word count
        for word,c in single_bow.items():
            if len(word) <100:
                WordCount_training.objects(word=word).update(upsert=True,inc__count=c)
                word_count_update[genre_word_update_template.format(word)]=c


        word_count_update["count"]=1
        #genre count
        db.GenreCount_training.update_one({"genre":training_obj.short_genre},
                                          {"$inc":word_count_update},
                                          upsert=True)



def _caculate_top_X_of_each_genre(top_x=1000):
    """
    Get the bow of each genre from GenreCount_training. Use mutual information calculation:
    P(f|c)P(c)log(N*P(f|c)/f)

    Since the comparison is intraclass, we can eliminate P(c), giving P(f|c)log(N*P(f|c)/f)

    Eventually, eliminating more terms, we get f_c * log(f_c * N / f), where f_c is the number of count of word f
        in class c

    To get relative measure of each word for each class.

    Top X of each genre is then chosen and stored in

    :param: top_x, top X word chosen from each category, default is 200
    :return:
    """
    print("Removing top word genre")
    TopWordGenre.objects().delete()
    print("Removing mutualinformationgenre")
    MutualInformationGenres.objects().delete()

    total_word_count=WordCount_training.objects().count()
    for c,genre_count_obj in enumerate(GenreCount_training.objects):
        print("Current at {}".format(genre_count_obj.genre))

        bow=genre_count_obj.bow

        mi_genre_dict={}
        for count,(word,word_freq_genre) in enumerate(bow.items()):
            if count%10000==0:
                print("Count is at {}".format(count))

            #calculate mu for each item
            word_count=WordCount_training.objects.get(word=word).count

            mi_genre_dict[word]=word_freq_genre*math.log(total_word_count*word_freq_genre/word_count)

        #sort and get top x
        sorted_list=itertools.islice(sorted(mi_genre_dict.items(),key=operator.itemgetter(1),reverse=True),0,top_x)

        #store the top mu
        TopWordGenre(genre=genre_count_obj.genre,bow=dict(sorted_list)).save()

        #just save the whole mu
        MutualInformationGenres(genre=genre_count_obj.genre,bow=mi_genre_dict).save()


def calculate_training_set_mu(training_coll_cls):
    assert training_coll_cls==TrainSetBow

    #_calc_word_genre_counts(training_coll_cls)
    _caculate_top_X_of_each_genre()


