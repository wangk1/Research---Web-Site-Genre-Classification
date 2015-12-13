import itertools,copy
from .util import *

from util.Logger import Logger
from data import Label

__author__ = 'Kevin'


data_logger=Logger()

"""
Label Convention

<cluster/classification>_


"""

class BaseData:
    """
    Base class for training and testing objects.

    A wrapper around the matrix that is the training or testing sets

    """

    def __init__(self,*,label,choose_ids,source,genre_mapping,pickle_dir,type="",vocab_vectorizer=None):
        """
        Source is an iterable adhering to the training and testing interface

        :param source:
        :return:
        """
        assert isinstance(label,Label)

        self.choose_ids=choose_ids
        self.source=source
        self.type=type

        self._X=None
        self._y=None
        self._ref_indexes=None

        self.vocab_vectorizer=vocab_vectorizer
        self.genre_mapping=genre_mapping
        self.label=label

        self.pickle_dir=pickle_dir

    def _load(self,*,stack_per_sample,pickle_X_path,pickle_y_path,pickle_ref_id_path,maybe_load_from_pickle,pickle):
        """
        Private method used to load both testing and training samples. Do not use

        If pickle file exists load the pickle file and X and y, else load from the self.source

        :param stack_per_sample:
        :param pickle_X_path:
        :param pickle_y_path:
        :param pickle_ref_id_path:
        :param maybe_load_from_pickle:
        :param pickle:
        :return:
        """

        assert hasattr(self,"vocab_vectorizer")

        loaded=False
        if maybe_load_from_pickle:
            data_logger.info("Trying to load {}'s {} samples from pickle".format(self.label,self.type))
            try:
                self._X=unpickle_obj(pickle_X_path)
                self._y=unpickle_obj(pickle_y_path)
                self._ref_indexes=unpickle_obj(pickle_ref_id_path)
                loaded=True
                data_logger.info("Successfully loaded samples from pickle")
            except FileNotFoundError:
                data_logger.info("Failed to load samples from pickle")

        if not loaded:
            if self.vocab_vectorizer is None:
                raise AttributeError("Vocab vectorizer does not exist")

            labels=[]
            matrix_cache=[]
            ref_indexes=[]
            for count,source_obj in enumerate(self.source):
                if count %1000==0:
                    data_logger.info("Train load curr at:  {}".format(count))

                curr_bow_matrix=self.vocab_vectorizer.transform(source_obj.attr_map)[0]
                matrix_cache.append(curr_bow_matrix)
                #genre mapping allows us to change the distribution of classes
                labels.append([self.genre_mapping.get(i,i) for i in source_obj.short_genre])
                ref_indexes.append(source_obj.ref_index)

                if len(matrix_cache)>stack_per_sample:
                    self._X=sp.vstack(matrix_cache)
                    matrix_cache=[self._X]
                    data_logger.info("stacked, train bow size:{},labels size: {}".format(
                        self._X.shape[0],len(labels)))

            if len(matrix_cache)>0:
                data_logger.info("stacking")
                self._X=sp.vstack(matrix_cache)
                del matrix_cache

            self._y=np.asarray(labels)
            self._ref_indexes=np.asarray(ref_indexes)

            #pickle if so
            if pickle:
                self.save_to_pickle(pickle_X_path,pickle_y_path,pickle_ref_id_path)

        data_logger.info("Final training size: {}".format(self._X.shape[0]))

    def save_to_pickle(self,pickle_X_path,pickle_y_path,pickle_ref_id_path):
        """
        Pickle the X,y, ref_id matrices with appropriate pickle path.

        :param pickle_X_path:
        :param pickle_y_path:
        :param pickle_ref_id_path:
        :return:
        """

        data_logger.info("Pickling {} set".format(self.type))
        pickle_obj(self._X,pickle_X_path)
        pickle_obj(self._y,pickle_y_path)
        pickle_obj(self._ref_indexes,pickle_ref_id_path)
        data_logger.info("Successfully save {} samples from pickle".format(self.type))


    def to_matrices(self):
        """
        Get matrices representation of X,y, and ref_indexes

        Filter here is used to filter out those rows whose ref_indexes is not in
            self._ref_indexes

        :return:
        """
        X=self._X
        y=self._y
        ref_indexes=self._ref_indexes

        if self.choose_ids:
            choices=np.apply_along_axis(lambda e:e in self.choose_ids,0,self._ref_indexes)

            X=X[choices]
            y=y[choices]
            ref_indexes=ref_indexes[choices]

        return X,y,ref_indexes

    @property
    def shape(self):
        return self._X.shape

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self,X):
        self._X=X


    @property
    def y(self):
        return self._y

    @y.setter
    def y(self,y):
        self._y=y

class Training(BaseData):
    def __init__(self,label,*,train_set_source=None,pickle_dir,genre_mapping=None,choose_ids=None):
        """
        :param label: Unique label
        :param pickle_dir: The main directory to leave all pickled vocabulary vectorizer
        :return:
        """
        super(Training, self).__init__(choose_ids=choose_ids,
                                       source=train_set_source,
                                       genre_mapping={} if not genre_mapping else genre_mapping,
                                       type="Training",label=label,pickle_dir=pickle_dir)


    def fit_vocab(self,*,load_vectorizer_from_file=True,pickle=True):
        """
        fit vocab to the attr_map of the training set implementing the data set interface. See init.py


        :return: None
        """

        vocab_loaded=False
        pickle_file=self.pickle_dir+"/vocab_{}_pickle".format(self.label.attr_select_technique)

        if load_vectorizer_from_file:
            try:
                self.vocab_vectorizer=unpickle_obj(pickle_file)

                vocab_loaded=True
                data_logger.info("Loaded pickle file {}".format(pickle_file))
            except FileNotFoundError:
                data_logger.info("Failed to find pickle file {}".format(pickle_file))
                vocab_loaded=False

        if not vocab_loaded:

            data_logger.info("Fitting vocab for classifier {}".format(self.label))

            self.vocab_vectorizer=DictVectorizer()

            words=[dict(itertools.chain(*(source_obj.attr_map.items() for source_obj in self.source)))]
            #fit the dv first
            self.vocab_vectorizer.fit(words)
            data_logger.info("vocab length is {}".format(len(self.vocab_vectorizer.feature_names_)))

            del words

            #pickle the obj if we want to
            pickle and pickle_obj(self.vocab_vectorizer,pickle_file)


        return self

    def load_training(self,stack_per_sample=3000,maybe_load_vectorizer_from_pickle=True,
                      maybe_load_training_from_pickle=True,pickle_training=True):
        """
        Load the training set with attr_map dictionary attribute and return a scipy sparse matrix of the data fitted
            with the vocab and their labels

        :returns: train_X: the data matrix. train_y: the train set labels

        """

        if self._X is not None:
            data_logger.info("Reloading training samples")

        if self.vocab_vectorizer is None:
            self.fit_vocab(load_vectorizer_from_file=maybe_load_vectorizer_from_pickle)

        data_logger.info("Loading training set for {}".format(self.label))

        trainX_pickle_path=self.pickle_dir+"/{}_trainX_{}_pickle".format(self.label.type,self.label.attr_select_technique)
        trainy_pickle_path=self.pickle_dir+"/{}_trainy_{}_pickle".format(self.label.type,self.label.attr_select_technique)
        ref_id_pickle_path=self.pickle_dir+"/{}_trainRefIndex_{}_pickle".format(self.label.type,self.label.attr_select_technique)

        self._load(stack_per_sample=stack_per_sample,
                  pickle_X_path=trainX_pickle_path,
                  pickle_y_path=trainy_pickle_path,
                  pickle_ref_id_path=ref_id_pickle_path,
                  maybe_load_from_pickle=maybe_load_training_from_pickle,
                  pickle=pickle_training)

        return self





class Testing(BaseData):
    def __init__(self,label,*,test_set_source=None,pickle_dir,genre_mapping=None,choose_ids=None,vocab_vectorizer=None):
        """
        :param label: Unique label
        :param pickle_dir: The main directory to leave all pickled vocabulary vectorizer
        :return:
        """
        super(Testing, self).__init__(choose_ids=choose_ids,
                                       source=test_set_source,
                                       genre_mapping={} if not genre_mapping else genre_mapping,
                                       type="Testing",label=label,pickle_dir=pickle_dir)

        self.vocab_vectorizer=vocab_vectorizer


    def load_testing(self,stack_per_sample=3000,pickle_testing=True,maybe_load_testing_from_pickle=True):
        """
        Load the training set with attr_map dictionary attribute and return a scipy sparse matrix of the data fitted
            with the vocab and their labels

        :returns: train_X: the data matrix. train_y: the train set labels

        """

        data_logger.info("Loading testing set for {}".format(self.label))

        testX_pickle_path=self.pickle_dir+"/{}_testX_{}_pickle".format(self.label.type,self.label.attr_select_technique)
        testy_pickle_path=self.pickle_dir+"/{}_testy_{}_pickle".format(self.label.type,self.label.attr_select_technique)
        ref_id_pickle_path=self.pickle_dir+"/{}_testRefIndex_{}_pickle".format(self.label.type,self.label.attr_select_technique)

        self._load(stack_per_sample=stack_per_sample,
                  pickle_X_path=testX_pickle_path,
                  pickle_y_path=testy_pickle_path,
                  pickle_ref_id_path=ref_id_pickle_path,
                  maybe_load_from_pickle=maybe_load_testing_from_pickle,

                  pickle=pickle_testing)

        return self