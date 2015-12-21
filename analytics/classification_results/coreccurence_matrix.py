__author__ = 'Kevin'

import collections as coll
import itertools as ittool
from tabulate import tabulate

from analytics.classification_results.res_iterator import RightResultsIter,WrongResultsIter

def _convert_to_list(inst):
    return [inst] if isinstance(inst,str) else inst

def coreccurence_matrix(right_res_iter,wrong_res_iter,do_print=True,heat_chart=True):
    assert isinstance(right_res_iter,RightResultsIter) and isinstance(wrong_res_iter,WrongResultsIter)

    actual_to_predicted=coll.defaultdict(lambda: coll.defaultdict(lambda:0))

    for right_res in right_res_iter:
        #get the right prediction
        predicted=right_res_iter.pred_transformer(right_res.predicted)
        actual=right_res_iter.actual_transformer(right_res.actual)

        predicted=_convert_to_list(predicted)

        actual=_convert_to_list(actual)

        for (a,p) in ittool.product(actual,predicted):
            actual_to_predicted[a][p]+=1

    for wrong_res in wrong_res_iter:
        predicted=wrong_res_iter.pred_transformer(wrong_res.predicted)
        actual=wrong_res_iter.actual_transformer(wrong_res.actual)

        predicted=_convert_to_list(predicted)

        actual=_convert_to_list(actual)

        for (a,p) in ittool.product(actual,predicted):
            actual_to_predicted[a][p]+=round(1/len(actual),3)

    if do_print:
        #print out tabular form of the matrix
        genres=sorted(actual_to_predicted.keys())

        column_headers=[""]+genres
        table_content=[[actual]+[actual_to_predicted[actual][predicted] for predicted in genres] for actual in genres]

        print(tabulate(table_content,headers=column_headers))

    return actual_to_predicted