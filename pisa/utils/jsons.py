#
# jsons.py
#
# A set of utilities for dealing with JSON files.
# Import json from this module everywhere (if you need to at all,
# and can not just use from_json, to_json)
# 
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

import os
import sys
import numpy as np
from pisa.utils.log import logging

#try and get the much faster simplejson if we can
try:
    import simplejson as json
    from simplejson import JSONDecodeError
    logging.trace("Using simplejson")
except ImportError:
    import json as json
    #No DecodeError in default json, dummy one
    class JSONDecodeError(ValueError):
      pass
    logging.trace("Using json")

def json_string(string):
    '''Decode a json string'''
    return json.loads(string)

def from_json(filename):
    '''Open a file in JSON format an parse the content'''
    try:
        content = json.load(open(os.path.expandvars(filename)),object_hook=NumpyDecoderHook)
        return content
    except (IOError, JSONDecodeError), e:
        logging.error("Unable to read JSON file \'%s\'"%filename)
        logging.error(e)
        sys.exit(1)

def to_json(content, filename,indent=2):
    '''Write content to a JSON file using a custom parser that
       automatically converts numpy arrays to lists.'''
    with open(filename,'w') as outfile:
        json.dump(content,outfile, cls=NumpyEncoder,
                  indent=indent, sort_keys=True)
        logging.debug('Wrote %.2f kBytes to %s'%
                  (outfile.tell()/1024.,os.path.basename(filename)))

class NumpyEncoder(json.JSONEncoder):
    """
    Encode to JSON converting numpy.ndarrays to lists
    """
    def default(self, o):

        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)

def NumpyDecoderHook(data):
    '''
    Object hook that converts lists to numpy arrays and unicode
    strings to python strings by iteratively walking through the data.
    (Only json returns "unicode", simplejson returns "str")
    '''

    #Use this same function to convert interatively
    convert = NumpyDecoderHook

    #Check all primitive types supported by json
    if isinstance(data, dict):
        return {convert(key): convert(value) for key, value in data.iteritems()}
    elif isinstance(data, list):
        try:
            #Any list that can go as float array should do so
            return np.array(data,dtype='float64')
        except ValueError:
            #all others are parsed iteratively
            return [convert(element) for element in data]
    elif isinstance(data, unicode):
        return data.encode('utf-8')
    else:
        return data
