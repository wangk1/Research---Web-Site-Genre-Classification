__author__ = 'Kevin'

from collections import namedtuple

ClassificationSource=namedtuple("Source",("ref_index","attr_map","short_genre"))

class ClassificationSourceMapper:
    """
    Used to create mapping from some iterable-> classification source
    """

    @staticmethod
    def __map(iter_obj,mapping):
        """
        Map each obj from iterable -> ClassificationSourceObj

        :param iter_obj:
        :param mapping:
        :return:
        """
        return ClassificationSource(ref_index=iter_obj[mapping["ref_index"]]
                                    ,attr_map=iter_obj[mapping["attr_map"]]
                                    ,short_genre=iter_obj[mapping["short_genre"]])

        pass

    @staticmethod
    def map(iterable,mapping):
        assert isinstance(mapping,dict)

        #reverse the value and keys for easy lookup
        mapping=dict((v,k) for k,v in mapping.items())

        return (ClassificationSourceMapper.__map(iter_obj,mapping) for iter_obj in iterable)