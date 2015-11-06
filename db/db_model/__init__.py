import operator
from mongoengine.base import ComplexBaseField
from mongoengine.common import _import_class
from warnings import warn

__author__ = 'Kevin'

"""
Contains the model files for the databases

"""


#Monkey patch

def to_python(self, value):
        """Convert a MongoDB-compatible type to a Python type.
        """
        Document = _import_class('Document')

        if isinstance(value, str):
            return value

        if hasattr(value, 'to_python'):
            return value.to_python()

        is_list = False
        if not hasattr(value, 'items'):
            try:
                is_list = True
                value = dict([(k, v) for k, v in enumerate(value)])
            except TypeError:  # Not iterable return the value
                return value

        if self.field:
            self.field._auto_dereference = self._auto_dereference
            value_dict = dict([(key, self.field.to_python(item))
                               for key, item in list(value.items()) if key != "__proto__"])
        else:
            value_dict = {}
            for k, v in list(value.items()):
                if isinstance(v, Document):
                    # We need the id from the saved object to create the DBRef
                    if v.pk is None:
                        self.error('You can only reference documents once they'
                                   ' have been saved to the database')
                    collection = v._get_collection_name()
                    from bson import DBRef
                    value_dict[k] = DBRef(collection, v.pk)
                elif hasattr(v, 'to_python'):
                    value_dict[k] = v.to_python()
                else:
                    value_dict[k] = self.to_python(v)

        if is_list:  # Convert back to a list
            return [v for k, v in sorted(list(value_dict.items()),
                                         key=operator.itemgetter(0))]
        return value_dict
warn("Note to self:Monkey Patching ComplexBaseField, considering removing")
"""
I had to monkey patch due to mongoengine's method not working and filtering out __proto__ in the bow dictionary and crashing after

trying to convert a dictionary to float
"""
ComplexBaseField.to_python=to_python