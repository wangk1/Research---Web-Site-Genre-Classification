__author__ = 'Kevin'


class MongoCollection:
    """
    Base class providing fundamental methods that represents mongodb collections

    This class can be used a decorator around the model to provide additional functionalities

    """

    #self.collection: name of the mongodb collection
    #self.query: kwargs of query params

    def __init__(self,collection_cls,**kwargs):
        self.collection_cls=collection_cls
        self.query=kwargs

    def __len__(self):
        """
        Gets the length of the collection

        :return: Length of collection
        """
        return len(self.collection_cls.objects)

    def __call__(self):
        return self

    def contains(self,**kwargs):
        return len(self.collection_cls.objects(**kwargs))>0

    def select(self,**kwargs):
        self.query=kwargs if kwargs is not None else {}

        return self

    def update(self,**kwargs):
        return self.find().update(**kwargs)

    def update_first(self,**kwargs):
        return self.find().update_one(**kwargs)

    def modify(self,**kwargs):
        """
        Similar to update, except upsert behaves a bit difference

        :param kwargs:
        :return:
        """
        return self.collection_cls.objects(self.query).modify(**kwargs)


    def find_one(self,index=0,**kwargs):
        res=self.find(**kwargs)

        res_len=len(res)

        return res[index if index<res_len else res_len-1] if res_len>0 else None

    def find(self,**kwargs):
        res=self.collection_cls.objects(**self.query)

        return res.only(*kwargs["only"]) if "only" in kwargs else res

    def remove(self):
        return self.find().delete()

    def create(self,save=True,**kwargs):
        doc_obj=self.collection_cls(**kwargs)

        return (save and doc_obj.save()) or doc_obj

    def __iter__(self):
        return self.collection_cls.objects

    def iterable(self):
        return self.collection_cls.objects