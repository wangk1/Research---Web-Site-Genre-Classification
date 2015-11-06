from sumy.summarizers._summarizer import SentenceInfo
from operator import attrgetter
from sumy.utils import *
from sumy.summarizers import AbstractSummarizer

__author__ = 'Kevin'

'''
Monkey patch prexisting libraries to provide behaviors we want

'''

__ALL__=['monkey_path']

MONKEY_PATCHED=False

def monkey_patch():
    global MONKEY_PATCHED

    if not MONKEY_PATCHED:
        #first and foremost, we want our own get best sentences behavior
        monkey_patch_AbstractSummarizer()
        MONKEY_PATCHED=True


"""
Monkey patch abstract summarizer so that it returns score also
"""
def monkey_patch_AbstractSummarizer():
    def _get_best_sentences(self, sentences, count, rating, *args, **kwargs):
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)

        # get `count` first best rated sentences
        # if not isinstance(count, ItemsCount):
         #    count = ItemsCount(count)
         #infos = count(infos)
        # # sort sentences by their order in document
        # infos = sorted(infos, key=attrgetter("order"))

        return infos

    AbstractSummarizer._get_best_sentences=_get_best_sentences;
