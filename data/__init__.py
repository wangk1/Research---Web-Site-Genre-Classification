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

    def __init__(self,type,dim_reduction,num_attributes,feature_selection,weights=tuple(),**kwargs):
        #DO NOT PUT ANYTHING UNLESS AFTER NOTED, WILL CAUSE INFINITE LOOP
        self.attr={"feature_selection":0,"dim_reduction":1,"num_attribute":2,"weight":3}
        self.attr_list=[]

        self.attr_list.append(feature_selection)
        self.attr_list.append(dim_reduction)
        self.attr_list.append(0)
        self.attr_list.append(weights)
        #SET VARIABLES ONLY AFTER THIS POINT
        self.type=type
        self.res_dir=kwargs.get("res_dir",None)
        self.kwargs=kwargs
        self.num_attributes=num_attributes

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

    @property
    def truncated_label(self):
        """
        Label that is truncated to 3 char per component

        :return:
        """
        return "_".join((str(i)[:4] for i in self.attr_list))