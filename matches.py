import numpy as np
from pathlib import Path
from CropImage import crop
from GetFeatures import Features
from PIL import Image
import cv2

def match(path):
    featuresDB = []
    img_pathsDB = []
    for feature_path in Path("./static/Features").glob("*.npy"):
        featuresDB.append(np.load(feature_path))
        img_pathsDB.append(Path("./static/Profile") / (feature_path.stem + ".jpg"))  
    featuresDB = np.array(featuresDB)
    img=cv2.imread(path)
    img=crop(img)
    save_path = 'static/CombinedFace/'
    save_path = save_path + "query.jpg" 
    cv2.imwrite(save_path, img)
    query=Features(save_path) 
    dists = np.linalg.norm(np.subtract(featuresDB,query), axis=1)
    ids = np.argsort(dists)[:3]  # Top 3 results
    scores = [(dists[id], img_pathsDB[id]) for id in ids]
    return scores