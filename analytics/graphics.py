__author__ = 'Kevin'
import matplotlib.pyplot as plt
import operator
import os,re

from scipy.stats import scoreatpercentile

PLOT_TOP_X=90
IGNORE_TOP_X=50
#stop words
#BUZZ_WORDS={"sentence","order","rating","sentenceinfo","www","http","com","https","us","aspx",'net'}
FIGURE=0

def plot_word_frequency(title,word_dictionary,plot_top=PLOT_TOP_X,reversed=False):
    """
    Plots a bar graph of a dictionary that contains words and its respective count

    Will only plot the top X
    :param word_dictionary:
    :return: plt obj, do whatever to it
    """
    words=sorted(word_dictionary.items(),key=operator.itemgetter(1),reverse=not reversed)[:plot_top]

    word_count=len(words)


    plt.title(title)
    plt.bar(range(word_count),[k[1] for k in words])
    plt.xticks(range(word_count),[k[0] for k in words],size= 6)
    locs,labels=plt.xticks()
    plt.setp(labels,rotation=90)

    plt.tight_layout()

    return plt

def save_fig(file_name,plt):
    folder_name="/".join(re.split("\\\\|/",file_name)[:-1])

    if not os.path.exists(folder_name):
        print("Creating folder: {}".format(folder_name))
        os.makedirs(folder_name)

    plt.savefig(file_name)

total_plot_number=4
fig_num=0
curr_figure=None
def subplot_four_corner(plt_num):
    """
    Automatically shape plt into a four plot per figure plot with each plot occupying four corners

    :param plt_num:
    :return:
    """

    global  fig_num
    global  total_plot_number
    global  curr_figure

    if plt_num%total_plot_number==0:
        fig_num+=1
        print("New figure number:{}".format(fig_num))

        curr_figure=plt.figure(fig_num)

    plt.subplot(2,2,(plt_num%4)+1)

    return plt_num==total_plot_number-1,curr_figure


def add_bar_plot(x,y):
    """
    Plot a single box plot of y values at coordinate x

    :param y: list of numbers
    :param x: x coord
    :return:
    """

    # percentiles of interest
    perc = [min(y), scoreatpercentile(y,10), scoreatpercentile(y,25),
                   scoreatpercentile(y,50), scoreatpercentile(y,75),
                   scoreatpercentile(y,90), max(y)]
    midpoint = x # time-series time

    # min/max
    plt.broken_barh([(midpoint-.01,.02)], (perc[0], perc[1]-perc[0]),edgecolor="k",facecolor="w")
    plt.broken_barh([(midpoint-.01,.02)], (perc[5], perc[6]-perc[5]),edgecolor="k",facecolor="w")
    # 10/90
    plt.broken_barh([(midpoint-.1,.2)], (perc[1], perc[2]-perc[1]),edgecolor="r",facecolor="w")
    plt.broken_barh([(midpoint-.1,.2)], (perc[4], perc[5]-perc[4]),edgecolor="r",facecolor="w")
    # 25/75
    plt.broken_barh([(midpoint-.4,.8)], (perc[2], perc[3]-perc[2]),edgecolor="b",facecolor="w")
    plt.broken_barh([(midpoint-.4,.8)], (perc[3], perc[4]-perc[3]),edgecolor="c",facecolor="w")

