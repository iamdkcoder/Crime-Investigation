import numpy as np
from PIL import Image
from Densenet import FeatureExtractor1
from efficientnetmod import FeatureExtractor2
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import cv2
from CropImage import crop
app = Flask(__name__)

# Read image features
fe1 = FeatureExtractor1()
fe2=FeatureExtractor2()
features = []
img_paths = []
for feature_path in Path("./static/feature1").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img1") / (feature_path.stem + ".jpg"))
  
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        print("file steam",file.stream)
        img = crop(uploaded_img_path)
        # path=str(img_path)

        path="static/croped/"+uploaded_img_path[18:]
        print("path",path)
        cv2.imwrite(path,img)
        # Run search
        queryA = fe1.extract(img=Image.open(path))
        queryB = fe2.extract(img=Image.open(path))
        query=np.concatenate((queryA,queryB))
        print(len(query))
        dists = np.linalg.norm(np.subtract(features,query), axis=1)  # L2 distances to features    
        ids = np.argsort(dists)[:4]  # Top 10 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
