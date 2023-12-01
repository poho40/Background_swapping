from flask import Flask, jsonify, request
import numpy as np
import os
import subprocess
app = Flask(__name__)


@app.route('/background')
def hello():
    print("hello")
    subprocess.call("python3 test_segmentation_deeplab.py -i colab_inputs/input", shell=True)
    subprocess.call("python3 test_pre_process.py -i colab_inputs/input", shell=True)
    output = subprocess.call("CUDA_VISIBLE_DEVICES=0 python3 test_background-matting_image.py -m real-fixed-cam -i colab_inputs/input/ -o colab_inputs/output/ -tb colab_inputs/background/0001.png", shell=True)
    print(output)
    return 'Hello, World!'

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))