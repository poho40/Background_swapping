from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
import subprocess
import shutil
from PIL import Image
import cv2
import matplotlib.pyplot as plt
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './iswbbb-frontend/public/background'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/background', methods=['POST'])
def hello():
    cards = request.get_json()['cards']
    images = []
    for card in cards:
        img = Image.open('/Users/rohit/Desktop/Umich2ndyear/Fall2023/EECS 442/442_Project/iswbbb-frontend/public' + card['url'])
        img=np.array(img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        x1, y1, height, width = int(card['dimensions']['x']), int(card['dimensions']['y']), int(card['dimensions']['height']), int(card['dimensions']['width'])
        print( x1, y1, height, width)
        blank_array = np.zeros((1920, 1080, 3), dtype=np.uint8)
        scaled_img = cv2.resize(img_bgr, (width, height))
        blank_array[y1:y1 + height, x1:x1 + width] = scaled_img
        images.append(blank_array)
    img_blend = np.zeros((1920, 1080,3))
    mask = np.ones_like(images[0], dtype=np.float32)
    for img in images:
        # Apply Gaussian blur to the alpha mask
        mask = cv2.GaussianBlur(mask, (0,0), 50)

        # Normalize the mask values to range [0, 1]
        mask = mask / 255.0
        img_blend = img_blend.astype(np.float32)
        img = img.astype(np.float32)
        # Blend the current image with the result using the mask
        img_blend = cv2.addWeighted(img_blend, 0.5, img, 0.5, 0)
    # for i in range(len(images)):     
    #     mask = np.zeros(images[i].shape)
    #     x1, y1, height, width = int(cards[i]['dimensions']['x']), int(cards[i]['dimensions']['y']), int(cards[i]['dimensions']['height']), int(cards[i]['dimensions']['width'])
    #     mask[y1:y1 + height, x1:x1 + width, :] = 1
    #     cv2.imwrite('mask_image.png', mask)
    #     img_blend = pyramid_blend(images[i].astype(float), img_blend, mask.astype(float), num_levels=4)
    #     img_blend=img_blend.astype(np.uint8)
    cv2.imwrite('output_image.png', img_blend)
    subprocess.call("CUDA_VISIBLE_DEVICES=0 python3 test_background-matting_image.py -m real-fixed-cam -i colab_inputs/input/ -o colab_inputs/output/ -tb output_image.png", shell=True)
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


def pyramid_upsample(img, kernel_size=(5,5)):
  """
  Upsamples the given pyramid image.
  Input:
    - img: an image of shape M x N x C
    - kernel_size: a tuple representing the shape of the 2D kernel
  Returns:
    - upsampled: an image represented as an array of shape 2M x 2N x C
  """
  #############################################################################
  # TODO: Implement pyramid upsampling.                                       #
  ###############################
  ##############################################
  new_img = np.insert(img, [i for i in range(len(img))], 0, axis=0)
  new_img = np.insert(new_img, [j for j in range(len(img[0]))], 0, axis=1)
  upsampled = 4*cv2.GaussianBlur(new_img, kernel_size, 1)
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return upsampled

def pyramid_downsample(img, kernel_size=(5,5)):
  """
  Downsamples the given pyramid image.
  Input:
    - img: an image of shape M x N x C
    - kernel_size: a tuple representing the shape of the 2D kernel
  Returns:
    - downsampled: an image of shape M/2 x N/2 x C
  """
  #############################################################################
  # TODO: Implement pyramid downsampling.                                     #
  #############################################################################
  new_img = cv2.GaussianBlur(img, kernel_size, 1)
  downsampled = np.delete(new_img , [i for i in range(0,len(img),2)], axis=0)
  downsampled = np.delete(downsampled, [j for j in range(0,len(img[0]),2)], axis=1)
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return downsampled

def gen_gaussian_pyramid(img, num_levels):
  """
  Generates an entire Gaussian pyramid.
  Input:
    - img: an image of shape M x N x C
    - num_levels: number of levels in the Gaussian pyramid
  Return:
    - gp: list, the generated levels (imgs) of the Gaussian pyramid
  """
  #############################################################################
  # TODO: Construct a Gaussian pyramid given a base image `img`.              #
  #############################################################################
  gp = []
  gp.append(img)
  new_img = img
  for i in range(num_levels - 1):
    new_img = pyramid_downsample(new_img)
    gp.append(new_img)
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return gp

def gen_laplacian_pyramid(gp, num_levels):
  """
  Generates an entire Laplacian pyramid.
  Input:
    gp: list, levels of a Gaussian pyramid
  Return:
    lp: list, the generated levels (imgs) of the Laplacian pyramid
  """
  #############################################################################
  # TODO: Construct a Laplacian pyramid given a base Gaussian pyramid `gp`.   #
  #############################################################################
  lp = []
  for i in range(num_levels - 1):
    lp.append(gp[i] - pyramid_upsample(gp[i+1]))
  lp.append(gp[len(gp)-1])
  lp.reverse()
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return lp

def reconstruct_img(lp):
  """
  Reconstructs an image using a laplacian pyramid.
  Input:
    lp: list, levels (imgs) of a Laplacian pyramid
  Return:
    recon_img: reconstructed image
  """
  recon_img = lp[0]
  for i in range(1, len(lp)):
    ###########################################################################
    # TODO: For each level, reconstruct the image from the Laplacian pyramid. #
    ###########################################################################
    recon_img = lp[i] + pyramid_upsample(recon_img)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

  return recon_img

def pyramid_blend(img1, img2, mask, num_levels=6):
  """
  This function produces the Laplacian pyramid blend of two images.
  Input:
    - img1: N x M x C uint8 array image
    - img2: N x M x C uint8 array image
    - mask: N x M array, all elements are either 0s or 1s
    - num_levels: int, height of the pyramid
  Return:
    - img_blend: the blended image, an N x M x C uint8 array
  """
  # Build Gaussian pyramids for img1, img2, and mask
  gp1, gp2, gpm = gen_gaussian_pyramid(img1, num_levels), gen_gaussian_pyramid(img2, num_levels), gen_gaussian_pyramid(mask, num_levels)
  # Build Laplacian pyramids for img1 and img2
  lp1, lp2 = gen_laplacian_pyramid(gp1, num_levels), gen_laplacian_pyramid(gp2, num_levels)
  #############################################################################
  # TODO: Construct the Laplacian pyramid and use it to blend the images.     #
  #############################################################################
  lp = []
  gpm.reverse()
  for i in range(num_levels):
    lp.append(lp1[i] * gpm[i] + lp2[i] * (1 - gpm[i]))
  img_blend = reconstruct_img(lp)
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return img_blend


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))