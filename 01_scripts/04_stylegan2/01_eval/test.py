import pickle
import numpy as np
import glob
import os
import sys
import PIL
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE
import scipy

arr = np.arange(15).reshape((3, 5))
np.random.shuffle(arr)
print(arr)