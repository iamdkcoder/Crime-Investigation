from flask import Flask, render_template, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from pathlib import Path
import os
import cv2
from datetime import datetime
import numpy as np
from CropImage import crop
from GetFeatures import Features
from matches import match
app = Flask(__name__)
app.secret_key = 'iamdk17'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/mydb'
db = SQLAlchemy(app)

class User(db.Model):
    name = db.Column(db.String(80), nullable=False)
    national_id = db.Column(db.String(80),primary_key=True , nullable=False)
    image_location = db.Column(db.String(120), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    birthmark = db.Column(db.String(120))

    def __repr__(self):
        return '<User %r>' % self.name



# Route for index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route for registerCriminal.html
@app.route('/registerCriminal',methods=['GET', 'POST'])
def registerCriminal():
    if request.method == 'POST':
        name = request.form['name']
        national_id = request.form['national_id']
        date_of_birth = request.form['date_of_birth']
        birthmark = request.form['birthmark']
        # Get the uploaded images
        img1 = request.files['front_view']
        img2 = request.files['left_view']
        img3 = request.files['right_view']
        # # # Save the front view image
        front_image_filename = name + '_' + national_id + '.jpg'
        front_image_path = os.path.join(app.root_path,'static', 'Profile', front_image_filename)

        # # Read the images using OpenCV

        img1 = cv2.imdecode(np.frombuffer(img1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img2.read(), np.uint8), cv2.IMREAD_COLOR)
        img3 = cv2.imdecode(np.frombuffer(img3.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Crop the face region from each image
        img = crop(img1)
        img = cv2.resize(img, (200, 200))
        img2 = crop(img2)
        img2 = cv2.resize(img2, (200, 200))
        img3 = crop(img3)
        img3 = cv2.resize(img3, (200, 200))

        #save the front face
        cv2.imwrite(front_image_path,img1)


        # Combine the images
        final_img = np.hstack((img2, img, img3))
        # final_img=cv2.resize(final_img,(224,224))
        # Save the final image
        save_path = 'static/CombinedFace/'
        save_path = save_path + front_image_filename
        cv2.imwrite(save_path, final_img)

        #save the features of combined image
        feature=Features(save_path)
        temp=name + '_' + national_id 
        feature_path = Path("./static/Features") / (temp + ".npy")
        np.save(feature_path,feature)
        img_path = 'static/profile/'+name+"_"+national_id+".jpg"
        user = User(name=name, national_id=national_id, image_location=img_path, date_of_birth=date_of_birth, birthmark=birthmark)
        db.session.add(user)
        db.session.commit()
        # Flash a success message
        flash('Form submitted successfully!','success')

        return render_template('registerCriminal.html')
    else:
         return render_template('registerCriminal.html')
    







# Route for findCriminal.html
@app.route('/findCriminal',methods=['GET', 'POST'])
def findCriminal():
    if request.method == 'POST':
        file=request.files['image']
        img=Image.open(file.stream)
        uploaded_img_path = "static/Query/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)        
        scores=match(uploaded_img_path)
        users=[]
        for score in scores:
            id = score[1].stem.split("_")[1]
            user = User.query.filter_by(national_id=id).first()
            users.append(user)
        return  render_template('findCriminal.html',users=users,query_path=uploaded_img_path,scores=scores)
    else:
        return render_template('findCriminal.html')

if __name__ == '__main__':
    app.run(debug=True)
