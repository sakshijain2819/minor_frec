from flask import Flask,render_template, request
import os
import Harcascade
import glob
import knn_test
import face_recognition

app = Flask(__name__)

res = os.path.abspath('./image1')
if glob.glob1(res, '*.jpg') != []:
   for fil in glob.glob1(res, '*.jpg'):
     print(os.path.join(res, fil))
     os.remove(os.path.join(res, fil))

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.basename('image1')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

namen = ''

@app.route('/')
def index():
    method1 = 'Harcascade'
    method2 = 'Convolution Neural Network'
    method3 = 'K_Nearest_Neighbour'
    return render_template('index.html',m1 = method1,m2 = method2, m3 = method3)

@app.route('/gen/Harcascade')
def generic(name='Harcascade'):
    namen=name
    return render_template('generic.html',name=name)

@app.route('/gen/K_Nearest_Neighbour')
def generic1(name='K_Nearest_Neighbour'):
    namen='K_Nearest_Neighbour'
    return render_template('generic1.html',name=name)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    img_name = file.filename
    f = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    file.save(f)
    Harcascade.fdec(f,img_name)
    res = os.path.abspath('./static/cropped')
    files = glob.glob1(res, '*.jpg')
    return render_template('generic.html',name=namen, img=file.filename, faces=files)

@app.route('/upload1', methods=['POST'])
def upload1():
  file = request.files['image']
  f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
  file.save(f)
  res = os.path.abspath('./static/imagek')
  if glob.glob1(res, '*.jpg') != []:
        for fil in glob.glob1(res, '*.jpg'):
            print(os.path.join(res, fil))
            os.remove(os.path.join(res, fil))


  for image_file in os.listdir('image1'):
        full_file_path = os.path.join('image1',image_file)
        print();
        print("Finding faces in {}".format(image_file))
        #print(pred);

        
       
        predictions = knn_test.predict(full_file_path,model_path="trained_knn_model.clf")
    

        for name, (top, right, bottom, left) in predictions:
            print("-  {} at ({}, {})".format(name, left, top))
           
            

        # Display results overlaid on an image
        knn_test.show_prediction_labels_on_image(os.path.join('image1', image_file), predictions,image_file)
        
        res = os.path.abspath('.static/imagek')
        files = glob.glob1(res, '*.jpg')
        return render_template('generic1.html',name=namen)


'''
def upload1():
  file = request.files['image']
  f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
  file.save(f)
  
  res = os.path.abspath('./static/cropped')

  if glob.glob1(res, '*.jpg') != []:
        for fil in glob.glob1(res, '*.jpg'):
            print(os.path.join(res, fil))
            os.remove(os.path.join(res, fil))
  
  for image_file in os.listdir('image1'):
        full_file_path = os.path.join('image1',image_file)
        print();
        print("Finding faces in {}".format(image_file))
        predictions = knn_test.predict(full_file_path,model_path="trained_knn_model.clf")
    

        for name, (top, right, bottom, left) in predictions:
            print("-  {} at ({}, {})".format(name, left, top))
           
            

        # Display results overlaid on an image
        knn_test.show_prediction_labels_on_image(os.path.join('image1', image_file), predictions,image_file)
        
        res = os.path.abspath('./static/imagek')
        files = glob.glob1(res, '*.jpg')
        return render_template('generic1.html',name=namen,img=file.filename,faces=files)
'''



if __name__ == '__main__':

    #app.run()
    #to run with python on server
    app.run(host='0.0.0.0',port =8888,debug=True)


