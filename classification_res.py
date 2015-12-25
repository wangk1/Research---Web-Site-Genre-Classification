__author__ = 'Kevin'
import collections as coll
import operator as op

from analytics.classification_results.res_iterator import RightResultsIter,WrongResultsIter
from analytics.classification_results.coreccurence_matrix import coreccurence_matrix


Recall=coll.namedtuple("Recall",("genre","true_pos","false_neg","recall"))
def recall(res_path,classifiers,secondary_identifier):
    recall=coll.Counter()
    total_counts=coll.Counter()
    for res_obj in RightResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier):
        recall.update(res_obj.actual)
        total_counts.update(res_obj.actual)

    for res_obj in WrongResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier):
        total_counts.update(res_obj.actual)

    recall=sorted([Recall(g[0],g[1],total_counts[g[0]],g[1]/total_counts[g[0]]) for g in recall.items()],key=op.itemgetter(3))
    print(recall)

    return recall

Precision=coll.namedtuple("Precision",("genre","true_pos","total","precision"))
def precision(res_path,classifiers,secondary_identifier):
    """
    Calculate the precision of the result. Given a result path folder and classifiers to look at.

    :param res_path:
    :param classifiers:
    :param secondary_identifier:
    :return:
    """
    precision=coll.Counter()
    total_counts=coll.Counter()

    right_res_iter=RightResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier)
    for res_obj in right_res_iter:
        precision.update(res_obj.actual)
        total_counts.update(res_obj.actual)

    wrong_res_iter=WrongResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier)
    for res_obj in wrong_res_iter:
        total_counts.update([wrong_res_iter.pred_transformer(res_obj.predicted)])

    precision=sorted([Precision(g[0],g[1],total_counts[g[0]],g[1]/total_counts[g[0]]) for g in precision.items()],key=op.itemgetter(3))
    print(precision)

    return precision

if __name__ == "__main__":
    res_path="classification_res\\supervised\\supervised_summary_chi_sq_10000"
    classifiers=["LogisticRegression"]
    secondary_identifier="no_region_kids"
    """
    right_res_iter=RightResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier)
    wrong_res_iter=WrongResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier)

    coreccurence_matrix(right_res_iter,wrong_res_iter)
"""
    rec=recall(res_path,classifiers,secondary_identifier)

    for rec_obj in rec:
        s="{}"

        attrs=[]
        for k,v in rec_obj._asdict().items():
            attrs.append(s.format(v))

        print(" ".join(attrs))
    print(len([i for i in RightResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier)]))
    print(len([i for i in RightResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier)]+
              [j for j in WrongResultsIter(result_path=res_path,classifier=classifiers, secondary_identifier=secondary_identifier)]))
