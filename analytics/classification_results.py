__author__ = 'Kevin'
import os,re,collections,operator

def get_wrong_sample(filepath):
    """
    Return a counter of test set and an indicator whether it was missed or not

    :return:
    """
    counter=None
    with open(filepath) as wrong_sample_file:

        counter=collections.Counter()
        for line in wrong_sample_file:

            res_list=line.split(" ")

            #counter.update([(res_list[0][:-1],res_list[2][:-1],res_list[4][:-1])])
            counter.update([res_list[0][:-1]])

    return counter

def count_miss_ratio():
    """
    Read from the result files and count the number of times each missed testing sample is missed out of the total classifiers

    :return:
    """
    path="C:\\Users\\Kevin\\Desktop\\GitHub\\Research\\Webscraper\\classification_res"
    
    wrong_file_list=[os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and re.match("^result.*wrong[.]txt$",f)]

    count=collections.Counter()

    for f in wrong_file_list:
        print(f)
        count+=get_wrong_sample(f)
    print(len(count))

    print(sorted(count.items(),key=operator.itemgetter(1),reverse=True))

