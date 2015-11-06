__author__ = 'Kevin'

from collections import namedtuple
import functools,math

from sumy.summarizers.lsa import LsaSummarizer as LSA
from sumy.summarizers.luhn import LuhnSummarizer as Luhn
from sumy.utils import get_stop_words

from util.text_preprocessor import *
from .summarizer_monkeypatch import monkey_patch
import summary.summarizer_settings as settings
import summary.summary_util as sutil
from sumy.nlp.stemmers import Stemmer

from nltk.corpus import stopwords
from nlp.stem import EnglishStemmer

LuhnSettings=namedtuple('LuhnSettings',('top_x_percent'))
LSASettings=namedtuple('LSASettings',())
SentenceInfo=namedtuple('SentenceInfo',('sentence','order','rating'))

INFINITY=-200

class Summarizer:
    black_list=[]

    def __init__(self):
        pass

    def luhn(self,text_parser):
        assert isinstance(text_parser,plaintext.PlaintextParser)

        summarizer=Luhn()
        #EnglishStemmer())
        #summarizer.stop_words=stopwords.words("english")

        summarizer.stop_words=get_stop_words(settings.SUMMARIZER_LANGUAGE)
        return summarizer(text_parser.document,settings.SUMMARIZER_TOP_X_SENTENCES)

    def lsa(self,text_parser):
        assert isinstance(text_parser,plaintext.PlaintextParser)

        #process the text
        summarizer=LSA()
        #EnglishStemmer())
        #summarizer.stop_words=stopwords.words("english")

        #we have to specify stop words
        summarizer.stop_words=get_stop_words(settings.SUMMARIZER_LANGUAGE)
        return summarizer(text_parser.document,settings.SUMMARIZER_TOP_X_SENTENCES)

    def __normalize_score(self,sentences_info):
        total_rating=functools.reduce(lambda acc,curr: acc+curr.rating,sentences_info,0)

        list_new_info=[]

        try:
            for sentence_info in sentences_info:
                list_new_info.append(SentenceInfo(sentence_info.sentence,sentence_info.order,math.log10(sentence_info.rating/total_rating)))
        except:
            #all good, just log having issues with small numbers
            pass

        return list_new_info

    def summarize(self,html_page,top_x_sentence=30):
        #we have to monkey patch to provide custom behavior we want
        monkey_patch()

        text_parser=preprocess_getParser(html_page)

        sentences_luhn=sorted(self.__normalize_score(self.luhn(text_parser)),key=lambda x:x.order)
        sentences_lsa=sorted(self.__normalize_score(self.lsa(text_parser)),key=lambda x:x.order)

        #combine values from each of the 4 summarization method for each sentence, note the order tells use which sentence
        #corresponds to the other, also we extract their scores
        # Each element of s_scores contains list of 4 additional element
        # 0-luhn setence score
        # 1-lsa score
        # 2-NB Score
        # 3-FOM score
        score_iters=(sutil.Lookahead(iter(sentences_luhn))
                     , sutil.Lookahead(iter(sentences_lsa)))

        sentence_scores_to_order={}
        for order in range(0,len(text_parser.document.sentences)):
            total_score=0
            match=0
            sentence=None
            #accumulate score from each summarization technique, if the setence does not exist, we just add infinity
            #offset to it
            for score_iter in score_iters:
                try:
                    next_sentence=score_iter.current()

                except StopIteration as IterStop:
                    next_sentence=None

                if next_sentence and next_sentence.order == order:
                    sentence=next_sentence
                    match+=1

                    total_score+=sentence.rating

                    #advance if we have a match
                    try:
                        score_iter.next()
                    except StopIteration:
                        pass

                else:
                    match=1
                    total_score+=INFINITY
            #no match in any cat
            if not sentence:
                sentence=SentenceInfo(text_parser.document.sentences[order],order,total_score)

            #push it backwards, we want to look up the order given the score
            sentence_scores_to_order[total_score/match]=(sentence.order if hasattr(sentence,"order") else order,sentence)

        return list(map(lambda s_tuple:str(s_tuple[1][1].sentence),sorted(sentence_scores_to_order.items(),key=lambda x:x[0],reverse=True)))[:top_x_sentence]
