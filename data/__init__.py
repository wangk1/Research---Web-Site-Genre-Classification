__author__ = 'Kevin'

import collections as coll

"""
Label used in classification and clustering pipeline

"""
class Label:

    def __init__(self,type,dim_reduction,num_feats,feature_selection):
        self.attr={"type":0,"attr_select_technique":1,"num_feats":3,"dim_reduction":2}
        self.attr_list=[]

        self.attr_list.append(type)
        self.attr_list.append(feature_selection)
        self.attr_list.append(dim_reduction)
        self.attr_list.append(num_feats)


    def __getattr__(self, item):
        return self.attr_list[self.attr[item]]

    def __setattr__(self, key, value):
        if key=="attr" or key=="attr_list":
            super().__setattr__(key,value)
            return

        if key not in self.attr:
            self.attr[key]=len(self.attr)
        self.attr_list.append(value)

    def __format__(self, format_spec):
        return self.__str__()

    def __str__(self):
        return "_".join((str(i) for i in self.attr_list))