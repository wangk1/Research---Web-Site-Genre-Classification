__author__ = 'Kevin'
import matplotlib.pyplot as plt
import operator

PLOT_TOP_X=90
IGNORE_TOP_X=50
#stop words
#BUZZ_WORDS={"sentence","order","rating","sentenceinfo","www","http","com","https","us","aspx",'net'}

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

    return plt

def save_fig(file_name,plt):
    plt.savefig(file_name)


