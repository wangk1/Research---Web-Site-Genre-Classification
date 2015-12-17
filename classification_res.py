__author__ = 'Kevin'
from classification.classification_res import RightResultsIter,WrongResultsIter
import collections as coll,operator as op

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

if __name__ == "__main__":
    res_path="classification_res\\supervised\\supervised_summary_chi_sq_10000"
    classifiers=["kNN"]
    secondary_identifier=""

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
