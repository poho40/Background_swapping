from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
import subprocess
import shutil
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import joblib

import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary

import albumentations as A
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './iswbbb-frontend/public/background'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/background', methods=['GET'])
def hello():
    query = "dogs in park"
    retrieve_images(query)
    # cards = request.get_json()['cards']
    # images = []
    # for card in cards:
    #     img = Image.open('/Users/rohit/Desktop/Umich2ndyear/Fall2023/EECS 442/442_Project/iswbbb-frontend/public' + card['url'])
    #     img=np.array(img)
    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     x1, y1, height, width = int(card['dimensions']['x']), int(card['dimensions']['y']), int(card['dimensions']['height']), int(card['dimensions']['width'])
    #     print( x1, y1, height, width)
    #     blank_array = np.zeros((1920, 1080, 3), dtype=np.uint8)
    #     scaled_img = cv2.resize(img_bgr, (width, height))
    #     blank_array[y1:y1 + height, x1:x1 + width] = scaled_img
    #     images.append(blank_array)
    # img_blend = np.zeros((1920, 1080,3))
    # for img in images:
    #     img_blend = img_blend.astype(np.float32)
    #     img = img.astype(np.float32)
    #     # Blend the current image with the result using the mask
    #     img_blend = cv2.addWeighted(img_blend, 0.5, img, 0.5, 0)
    # for i in range(len(images)):     
    #     mask = np.zeros(images[i].shape)
    #     x1, y1, height, width = int(cards[i]['dimensions']['x']), int(cards[i]['dimensions']['y']), int(cards[i]['dimensions']['height']), int(cards[i]['dimensions']['width'])
    #     mask[y1:y1 + height, x1:x1 + width, :] = 1
    #     cv2.imwrite('mask_image.png', mask)
    #     img_blend = pyramid_blend(images[i].astype(float), img_blend, mask.astype(float), num_levels=4)
    #     img_blend=img_blend.astype(np.uint8)
    # cv2.imwrite('output_image.png', img_blend)
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
        # if 'files[]' not in request.files:
        #     return jsonify({'error': 'No files part in the request'}), 400

        text = request.form.get('text')
        image = request.files.get('image')
        back = request.files.get('back')
        print(request.files)
        uploaded_files = []
        # for file in files:
        #     if file:
        #         # Save the file to the UPLOAD_FOLDER
        #         filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        #         file.save(filename)
        #         uploaded_files.append(filename)
        
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
        if text:
            print(text)
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

    
def retrieve_images(query):
    image_embeddings = torch.load('image_embeddings.pt', map_location=torch.device('cpu'))
    with open('image_filenames', 'rb') as file:
    # Load the data from the file
      image_filenames = pickle.load(file)
    model = torch.load('flickr8.pt', map_location=torch.device('cpu'))

    tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)


    ##############################################################################
    #                               YOUR CODE HERE                               #
    ##############################################################################
    # TODO: Please normalize the image_embeddings and text_embeddings using L2norm,
    # calculate the similarity,
    # and then get the top k similar images to the given text query and pass them to the 'matches' variable.
    # Note: You cannot use any for loops in this function.
    ##############################################################################
    matches = None # Please keep this as the variable name for the matches
    image_path = "/content/Images"
    img_norm = F.normalize(image_embeddings,p=2, dim=-1)
    text_norm = F.normalize(text_embeddings, p=2, dim=-1)
    normal = text_norm @ img_norm.T
    vals, indices = torch.topk(normal.squeeze(0), 1)
    matches = image_filenames[torch.Tensor.tolist(indices)]
    print(matches)
    image = cv2.imread(f"{image_path}/{matches[0]}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output_image.png', image)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

image_path = "/content/Images"
captions_path = "/content"
batch_size = 32
num_workers = 2
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 1
factor = 0.8
epochs = 3

image_encoder_model = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 0.07

image_size = 64

# for projection head; used for both image and text encoders
projection_dim = 256

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)

# data augmentation for images
def get_transforms(mode="train"):
    return A.Compose(
            [
                A.Resize(image_size, image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

from torchvision.models.resnet import ResNet50_Weights
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=image_encoder_model, pretrained=pretrained, trainable=trainable
    ):
        super().__init__()
        self.model = torchvision.models.resnet50(ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

class TextEncoder(nn.Module):
    def __init__(self, model_name=text_encoder_model, pretrained=pretrained, trainable=trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=projection_dim,
    ):
        super().__init__()
        '''
        Args:
            embedding_dim (int): Extracted Image or text feature embedding dimenasion.
        '''

        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Initialize a single layer linear transformation for the projection   #
        # head.                                                                      #
        ##############################################################################
        self.linear = nn.Linear(embedding_dim, projection_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, x):
        '''
        Args:
            x: Image or text feature embeddings extracted from the ResNet50 and DistilBERT model respectively.

        Return:
            projected: The projected image and text embeddings.
        '''
        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Write the forward function. Normalize the output of the projection   #
        # head. Hint: use F.normalize() for the normalization                        #
        ##############################################################################

        projected = self.linear(x)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return projected
    
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=temperature,
        image_embedding=image_embedding,
        text_embedding=text_embedding,
    ):
        super().__init__()
        '''
        Args:
            temperature (float): temperature parameter which controls the range of the logits.
            image_embedding (int): Shape of the extracted image embedding
            text_embedding (int): Shape of the extracted text embedding

        '''

        self.image_encoder = None
        self.text_encoder = None
        self.image_projection = None
        self.text_projection = None
        self.temperature = temperature
        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Initialize the encoders and the projection heads for image and text i.e,
        # instantiate the above None variables with their corresponding models       #
        ##############################################################################
        self.image_encoder = ImageEncoder(image_encoder_model, pretrained=True, trainable=True)
        self.text_encoder = TextEncoder(text_encoder_model, pretrained=True, trainable=True)
        self.image_projection = ProjectionHead(image_embedding, projection_dim)
        self.text_projection = ProjectionHead(text_embedding, projection_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################


    def forward(self, batch):

        '''
        Args:
            batch: batch of images for training.

        Return:
            loss: computed loss.
        '''
        # get image and text features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        loss = None
        ##############################################################################
        #                               YOUR CODE HERE                               #
        ##############################################################################
        # TODO: Project image_features and text_features into a new vector space and write the loss function by following the above equations
        # Hint: You can make use of nn.CrossEntropyLoss() or nn.LogSoftmax() when calculating the loss
        # you are not allowed to use any for loops while computing the loss.
        ##############################################################################
        criterion = nn.CrossEntropyLoss()
        im_projection = F.normalize(self.image_projection(image_features))
        te_projection = F.normalize(self.text_projection(text_features))
        logits = (te_projection @ im_projection.T) / self.temperature
        text_loss = criterion(logits, torch.arange(batch_size))
        image_loss = criterion(logits.T, torch.arange(batch_size).T)
        loss = (text_loss + image_loss)/2
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return loss.mean()
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))