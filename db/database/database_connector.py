__author__ = 'Kevin'

class DBClient:
    #self.host_name
    #self.port

    #self.connection

    def __init__(self,host_name,port):
        self.host_name=host_name
        self.port=port
        self.connection=None

    #override but should call
    def connect(self):
        raise NotImplementedError("Not implemented")


