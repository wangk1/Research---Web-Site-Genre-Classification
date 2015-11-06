__author__ = 'Kevin'

from .Logger import Logger
from .exception_handler import *

__all__=['fail_safe_with_handler','fail_safe_mongo']

#Predefined error handlers
__mongo_handler=MongoHandler()
__web_handler=WebHandler()
__default_handler=DefaulExceptionHandler()

"""
This is a function decorator that will catch any error, preventing premature termination.

The errors will be logged

"""
def fail_safe_with_handler(exc_handler):

    def fail_safe_handler_caller(func):
        def fail_safe_wrapper(*args,**kwargs):

            try:
                return func(*args,**kwargs)

            except Exception as ex:
                Logger.error('Exception occured: {}'.format(str(ex)))

                exc_handler.handle(ex)

            return None

        return fail_safe_wrapper

    return fail_safe_handler_caller

def fail_safe(func):
    return fail_safe_with_handler(__default_handler)(func)

"""
Annotation, handles mongo db failures
"""
def fail_safe_mongo(func):
    return fail_safe_with_handler(__mongo_handler)(func)

def fail_safe_web(func):
    return fail_safe_with_handler(__web_handler)(func)
