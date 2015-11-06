from sumy.summarizers._summarizer import AbstractSummarizer

from bs4 import BeautifulSoup
from collections import Counter
import itertools,functools

from sklearn.metrics.pairwise import cosine_similarity as cos_sim

__author__ = 'Kevin'


class NaiveBayesSummarizer(AbstractSummarizer):
    #self._documents

    def __init__(self):
        pass

    def __call__(self, document,html_document):
        pass

    def train(self,documents,htmls_documents):
        """
        Train the naive bayes classifier with input documents and html_documents
        :param documents:
        :param htmls_documents:
        :return:
        """
        pass

    def __generate_attributes(self,document):
        pass

    @staticmethod
    def sentence_position(sentence,document):
        """
        Attribute 1, position of sentence in paragraph
        """
        for p in document.paragraphs:
            for loc,s in enumerate(p.sentences):
                if s==sentence:
                    return loc


    @staticmethod
    def sentence_length(sentence):
        """
        Attribute 2: length of sentence

        """
        return len(sentence.words)


    @staticmethod
    def word_importance(sentence,curr_doc,*documents):
        """
        Attribute 3: Sum of (TF of word * num sentence each word in) for each document
        """
        #gets us word frequency in the current document
        word_freq=Counter(curr_doc)

        #gets us sentence frequency in the b document pages
        sentence_freq=Counter()
        for sentence in itertools.chain(doc.sentences for doc in documents):
            sentence_freq+=Counter(set(sentence.words))

        #for each word in the current sentence, get its term freq across all document times the number of occurence it appears in sentences.
        # and sum them up
        return functools.reduce(lambda acc,word: sentence_freq.get(word,0)*word_freq.get(word,0)+acc,sentence.words,0)

    @staticmethod
    def sentence_similarity_with_all_sentence(sentence_bow,document_bow):
        """
        Attribute 5: Similarity between each sentence and all the other sentences in the model

        Difference between sentence s and all text in the page

        First, convert the sentence and target sentence to TF-IDF.

        Then, get the cosine similarity.
        :param sentence_bow: bag of word of the current sentence under consideration
        :param document_bow:bag of words of all other setences in the document
        :return:
        """

        s_cosine_sim=functools.reduce(lambda mean,curr_other_sentence:
                         mean+cos_sim(sentence_bow,curr_other_sentence)/len(document_bow),
                         document_bow,0)
        return s_cosine_sim


    @staticmethod
    def words_in_special_word_set(sentences,html_doc):
        """
        Attribute 7: Occurence of word Si in special word set. Aka, words that are italicized, bolded or underlined
        :param sentence: NOTE this is a sentence representation, NOT BOW
        :param html_doc: HTML page
        :return:
        """
        html_doc=BeautifulSoup(html_doc,'html.parser')
        italicized=set([i_tag.string for i_tag in html_doc.find_all('i')])
        bolded=set([i_tag.string for i_tag in html_doc.find_all('b')])
        underlined=set([i_tag.string for i_tag in html_doc.find_all('u')])

        important_words=itertools.chain(italicized,bolded,underlined)

        def score(sentence):
            return functools.reduce(lambda sum,curr_word:sum+(curr_word in important_words),sentence.words,0)

        #maps respective sentence to their scores
        return itertools.starmap(score,sentences)

    @staticmethod
    def average_font_size():
        pass
