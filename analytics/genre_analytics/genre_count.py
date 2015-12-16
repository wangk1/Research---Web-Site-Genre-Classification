from data.util import unpickle_obj
from util.base_util import normalize_genre_string
from analytics.graphics import add_bar_plot
from data import bad_genre_set

import matplotlib.pyplot as plt
import operator as op

import collections as coll

__author__ = 'Kevin'


def num_genre_per_webpage(matrix_path):
    """
    Create a box plot of how many other genres each webpage has for each genre

    Also, record the occurence of genres with each other
    :param matrix_path:
    :return:
    """

    label_matrix=unpickle_obj(matrix_path)

    genre_to_num_webpages=coll.defaultdict(lambda:[])

    for webpage_genre in label_matrix:

        normalized_genre=set([normalize_genre_string(g,1) for g in webpage_genre])

        for g in normalized_genre:
            if g in bad_genre_set:
                continue

            #if normalized_genre-{g}:
            genre_to_num_webpages[g].append(normalized_genre-{g})


    #box plot it
    genre_to_num_item_iter=genre_to_num_webpages.items()

    plt.clf()
    plt.figure(1)

    plt.xticks([i for i in range(0,len(genre_to_num_item_iter))],[op.itemgetter(0)(i) for i in genre_to_num_item_iter])
    plt.yticks(range(0,6))
    plt.tick_params(axis="both",which="major",labelsize=5)

    for c,(g,counts) in enumerate(genre_to_num_item_iter):
        add_bar_plot(c,[ len(gs) for gs in counts])

    plt.savefig("C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res\\genre_analysis\\genre_dist.pdf")
    #print
    print(genre_to_num_webpages)
