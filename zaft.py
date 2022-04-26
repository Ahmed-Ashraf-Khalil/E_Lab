# import libary
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import img_to_array 
from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import io


#Mapping Labels

all_labels = ['Atelectasis',
 'Cardiomegaly',
 'Consolidation',
 'Edema',
 'Effusion',
 'Emphysema',
 'Fibrosis',
 'Infiltration',
 'Mass',
 'Nodule',
 'Pleural_Thickening',
 'Pneumonia',
 'Pneumothorax']

# initialize our Flask application and the Keras model
app= Flask(__name__,template_folder='templates')
model=None


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # substitute in your own networks just as easily)
    global model_Xray
    model_Xray =tf.keras.models.load_model('D:\\E-lab Project\\Website\\x-ray_chest87.sav')


def prepare_image_xray(image_x, target):
    # if the image mode is not RGB, convert it
    
    image_x = image_x.resize(target)
    image_x = image_x.convert("L")
    image_x = img_to_array(image_x)
    image_x = np.expand_dims(image_x, axis=0)
    

    # return the processed image
    return image_x

#upload_folder="D:\\Test Api\\Static\\"

@app.route("/", methods=['GET', 'POST'])
def Skin_Cancer():
	return render_template("xray.html")

@app.route('/XRAY', methods=["GET","POST"])
def XRAY():
   # initialize the data dictionary that will be returned from the
   # view
   data = {"success": False}
   
   # ensure an image was properly uploaded to our endpoint
   if request.method == "POST":
        if  request.files["image"]:
            # read the image in PIL format
            image_x = request.files["image"].read()
            image_x = Image.open(io.BytesIO(image_x))

            # preprocess the image and prepare it for classification

            image_x = prepare_image_xray(image_x, target=(128, 128))
            
            #with sess.as_default():
            #with graph.as_default():
            preds = model_Xray.predict(image_x)
            data["pre"] = []

            l={}
            for n_class, p_score in zip(all_labels, preds):
                l[n_class[:]]=p_score*100
            
            results = max(l,key=l.get)
            data["pre"].append(results)
            # returned prediction
            

            # indicate that the request was a success
            data["success"] = True
    
        return render_template('xray.html',pre=results)
    # return the data dictionary as a JSON response
   return flask.jsonify(data)
    
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(debug=True,use_reloader=False)