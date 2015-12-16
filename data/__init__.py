__author__ = 'Kevin'

import collections as coll,copy

bad_genre_set={"Kids_and_Teens","genres"}

"""
Label used in classification and clustering pipeline

"""
class LearningSettings:
    """
    Label used in classification and clustering pipelines to store critical informations.

    MUST BE deep copyable

    """

    def __init__(self,type,dim_reduction,num_feats,feature_selection,**kwargs):
        #DO NOT PUT ANYTHING UNLESS AFTER NOTED, WILL CAUSE INFINITE LOOP
        self.attr={"type":0,"feature_selection":1,"num_feats":3,"dim_reduction":2}
        self.attr_list=[]

        self.attr_list.append(type)
        self.attr_list.append(feature_selection)
        self.attr_list.append(dim_reduction)
        self.attr_list.append(num_feats)
        #SET VARIABLES ONLY AFTER THIS POINT

        self.res_dir=kwargs.get("res_dir",None)
        self.kwargs=kwargs

        for k,v in kwargs.items():
            setattr(self,k,v)

    def __deepcopy__(self,memo):
        attr_copy=copy.deepcopy(self.kwargs)
        attr_copy.update({k:self.attr_list[v] for k,v in self.attr.items()})

        return LearningSettings(**attr_copy)

    def __getattr__(self, item):
        return self.attr_list[self.attr[item]]

    def __setattr__(self, key, value):
        if key=="attr" or key=="attr_list" or key not in self.attr:
            super().__setattr__(key,value)
            return

        self.attr_list[self.attr[key]]=value

    def __format__(self, format_spec):
        return self.__str__()

    def __str__(self):
        return "_".join((str(i) for i in self.attr_list))

    @property
    def label(self):
        return str(self)