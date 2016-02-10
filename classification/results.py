import operator,statistics

__author__ = 'Kevin'
import collections as coll


ResSingle=coll.namedtuple("SingleResult",("accuracy","weights","classifier_names"))
CVResult=coll.namedtuple("CVResult",("avg_accuracy","std_dev","weights","classifier_names"))

class ResCrossValidation:

    def __init__(self):
        self._results=coll.defaultdict(lambda:[])


    def update(self,*res_singles,kth_fold):
        for res_single in res_singles:
            assert isinstance(res_single,ResSingle)
            key=(res_single.classifier_names,res_single.weights)

            if kth_fold == len(self._results[key]):
                self._results[key].append(res_single)
            else:
                self._results[key][kth_fold]=res_single

    @property
    def results(self):
        result=[]
        for res_key in self._results.keys():
            assert all((res_single.weights==self._results[res_key][0].weights and res_single.classifier_names==self._results[res_key][0].classifier_names
                       for fold, res_single in enumerate(self._results[res_key])))

            all_acc=[res_single.accuracy for fold, res_single in enumerate(self._results[res_key])]

            res_single=self._results[res_key][0]
            result.append((statistics.mean(all_acc),statistics.stdev(all_acc),res_single.weights,res_single.classifier_names))

        return result

    @property
    def best_result(self):
        #average the folds
        #TODO:FIX THIS ADD IN FOLD INFO

        return sorted(self._results.items(),key=operator.itemgetter(1),reverse=True)[0]