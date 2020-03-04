import time
import numpy as np
import _pickle as cPickle
def save_pkl(obj, path):
  with open(path, 'wb') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)
def load_pkl(path):
  with open(path, 'rb') as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj