from flask import Flask,render_template,request,url_for
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras.models import Sequential
'''


img_height = 224
img_width = 224


def loadModel(predict_model):
    test_path = (image_path)
    img = keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = predict_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score

#Load_model2
K =["ResNet50_GCD","ResNet50_CV","ResNet50_CV_Battery","ResNet50_CV_Pseudo"]
m=1
for model in K :
    m+=1
    Model =model
    filepath=f'model1_ResNet50.h5'
    filepath_model = f'model1_ResNet50.json'
    filepath_weights = f'weights_model_{Model}.h5'
    #Load

    if m ==2 :
        predict_model_2 = load_model(filepath)
        with open(filepath_model, 'r') as f:
            loaded_model_json = f.read()
            predict_model_2 = model_from_json(loaded_model_json)
            predict_model_2.load_weights(filepath_weights)
    elif m==3 :
        predict_model_3 = load_model(filepath)
        with open(filepath_model, 'r') as f:
            loaded_model_json = f.read()
            predict_model_3 = model_from_json(loaded_model_json)
            predict_model_3.load_weights(filepath_weights)
    elif m==4 :
        predict_model_4 = load_model(filepath)
        with open(filepath_model, 'r') as f:
            loaded_model_json = f.read()
            predict_model_4 = model_from_json(loaded_model_json)
            predict_model_4.load_weights(filepath_weights)
    elif m ==5 :
        predict_model_5 = load_model(filepath)
        with open(filepath_model, 'r') as f:
            loaded_model_json = f.read()
            predict_model_5 = model_from_json(loaded_model_json)
            predict_model_5.load_weights(filepath_weights)
'''

app = Flask(__name__)


@app.route('/',methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/GCD',methods=['GET'])
def GCD():
    return render_template("GCD.html")
'''
@app.route('/GCD',methods=['POST'])
def predict():
    global image_path
    
    imagefile = request.files['imagefile_GCD']
    #image_path= "./static/"+imagefile.filename
    image_path= "./static/"+"imagefile.png"
    imagefile.save(image_path)
    score= loadModel(predict_model_2)
    #Battery 100%
    if score[0]==np.max(score) : 
        PictureExt = 'Battery'
        score_r=score[0]
        score_a =f"{score_r/75*100*100:.2f}%"
    #Pseudocapacitor  
    elif score[1]==np.max(score) :
        PictureExt = "Pseudocapacitor"
        score_r=score[1]
        score_a =f"{score_r*100/75*100:.2f}%"
    classification = {"predict":PictureExt,"score":score_a}
    return render_template("GCD.html", prediction=classification,image_path=image_path)
'''
@app.route('/CV',methods=['GET'])
def CV():
    return render_template("CV.html")
'''

@app.route('/CV',methods=['POST'])
def predict2(): 
    global image_path,predict_model
    imagefile = request.files['imagefile_CV']
    #image_path= "./static/"+imagefile.filename
    image_path= "./static/"+"imagefile.png"
    imagefile.save(image_path)
    score= loadModel(predict_model_3)

    #Battery 100%
    if score[0]==np.max(score) : 
        PictureExt = 'Battery'
        score_r=score[0]
        score_a =f"{score_r*100:.2f}%"
        score= loadModel(predict_model_4)
        if score[0]==np.max(score) :
            PictureP = (1-score[0]/75*100)
            score_P =f"{PictureP*100:.2f}% "  
            PictureN= "0"
        if score[1]==np.max(score) :
            PictureP = 0.5*((score[1]/75*100))
            score_P =f"{PictureP*100:.2f}% " 
            PictureN= "1"            
    #Pseudocapacitor  
    elif score[1]==np.max(score) :
        PictureExt = "Pseudocapacitor"
        score_r=score[1]
        score_a =f"{score_r*100:.2f}%"
        score= loadModel(predict_model_5)
        if score[0]==np.max(score) :
            PictureP = score[0]/75*100
            score_P =f"{PictureP*100:.2f}%"   
            PictureN= "0"
        if score[1]==np.max(score) :
            PictureP = 0.5*(1-(score[1]/75*100)+1)
            score_P =f"{PictureP*100:.2f}%" 
            PictureN= "1"                    



    classification = {"predict":PictureExt,"score":score_a,"psedo":score_P}
    return render_template("CV.html", prediction2=classification,image_path=image_path)
'''



@app.route('/about')
def about():
    products = ["เสื้อผ้า","เตารีด","ผ้าห่ม"]
    return render_template("about.html",myproduct=products)

@app.route('/admin')
def profile():
    #name age
    username = "Konthee"
    age = 26
    return render_template("admin.html",username=username,myage=age)

@app.route('/sendData')
def signupForm():
    fname=request.args.get('fname')
    description= request.args.get('description')
    return render_template("thankyou.html",data={"name":fname,"description":description})

@app.route('/Contact')
def Contact():
    
    return render_template("Contact.html")


if __name__== "__main__":
    app.run(debug=True)


