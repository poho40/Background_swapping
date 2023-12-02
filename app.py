from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
import subprocess
import shutil
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './iswbbb-frontend/public/background'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/background')
def hello():
    subprocess.call("python3 test_segmentation_deeplab.py -i colab_inputs/input", shell=True)
    subprocess.call("python3 test_pre_process.py -i colab_inputs/input", shell=True)
    output = subprocess.call("CUDA_VISIBLE_DEVICES=0 python3 test_background-matting_image.py -m real-fixed-cam -i colab_inputs/input/ -o colab_inputs/output/ -tb colab_inputs/background/0001.png", shell=True)
    return 'Hello, World!'

@app.route('/upload-multiple', methods=['POST'])
def upload_multiple_files():
    shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])
    shutil.rmtree('./colab_inputs/input/')
    os.makedirs('./colab_inputs/input/')
    try:
        # Check if the post request has the file part
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files part in the request'}), 400

        files = request.files.getlist('files[]')
        image = request.files.get('image')
        back = request.files.get('back')
        print(request.files)
        uploaded_files = []
        for file in files:
            if file:
                # Save the file to the UPLOAD_FOLDER
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                uploaded_files.append(filename)
        
        if image:
            print(image)
            image_filename = os.path.join('./colab_inputs/input/', '442_img.png')
            print(image_filename)
            image.save(image_filename)
        # Process background
        if back:
            back_filename = os.path.join('./colab_inputs/input/', '442_back.png')
            print(back_filename)
            back.save(back_filename)
        subprocess.call("python3 test_segmentation_deeplab.py -i colab_inputs/input", shell=True)
        subprocess.call("python3 test_pre_process.py -i colab_inputs/input", shell=True)
        return jsonify({'message': 'Files uploaded successfully', 'files': uploaded_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/getnames')
def get_names():
    path = os.getcwd() + "/iswbbb-frontend/public/background/"
    uploaded_files = []
    for filename in os.listdir(path):
        if(filename[-3:] == "png"):
            uploaded_files.append(filename)
    return jsonify({'files': uploaded_files})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))