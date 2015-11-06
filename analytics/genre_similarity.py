__author__ = 'Kevin'

from util.base_util import normalize_genre_string

def get_genre_similarities():

    #the threshold where the two categories are similar and when they are different
    THRESHOLD_SIMILAR=0.5
    THRESHOLD_DIFF=0.2
    '''
    sim_sam_cat->very similar genres derived from the same parent
    sim_diff_cat->very similar genres derived from different parents
    diff_sim_cat->very different genres derived from the same parent
    '''

    with open("genre_similarity.txt",encoding="latin-1") as sim_txt \
        ,open("genre_similarity_similar_cat.txt",encoding="latin-1",errors="ignore",mode="a") as sim_sam_cat \
        ,open("genre_similarity_similar_diff_cat.txt",encoding="latin-1",errors="ignore",mode="a") as sim_diff_cat \
        ,open("genre_similarity_diff_similar_cat.txt",encoding="latin-1",errors="ignore",mode="a") as diff_sim_cat:

        #line format:genre1, genre2 value: num\n
        for line in sim_txt:
            split_line=line.split(" ")

            genre1=normalize_genre_string(split_line[0][:-1])
            #the :
            genre2=normalize_genre_string(split_line[1])

            sim_score=float(split_line[3][:-1])

            #If the two genres are similar or different
            if sim_score>THRESHOLD_SIMILAR:
                if has_same_parent(genre1,genre2):
                    sim_sam_cat.write("{}".format(line))
                else:
                    sim_diff_cat.write("{}".format(line))

            elif sim_score<THRESHOLD_DIFF:
                if has_same_parent(genre1,genre2):
                    diff_sim_cat.write("{}".format(line))


def has_same_parent(g1,g2):
    return g1.split("/")[0] == g2.split("/")[0]
