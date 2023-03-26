from FeatureExtractor import DenseNetFeatures
from FeatureExtractor import EfficientNetFeatures
from PIL import Image
import numpy as np

fe1=DenseNetFeatures()
fe2=EfficientNetFeatures()
def Features(path):
    queryA=fe1.extract(img= Image.open(path))
    queryB=fe2.extract(img= Image.open(path))
    concanateFeature = np.concatenate((queryA,queryB))
    return concanateFeature