import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary

import albumentations as A
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from tqdm import tqdm_notebook
from tqdm.autonotebook import tqdm
from torchvision.models.resnet import ResNet50_Weights
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
feat_dim = 2048
unlabeled_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(64),
                                         transforms.ToTensor()])

labeled_transform = transforms.Compose([transforms.CenterCrop(64),
                                        transforms.ToTensor(),])


# We use the PyTorch built-in class to download the STL-10 dataset.
# The 'unlabeled' partition contains 100,000 images without labels.
# It's used for leanring representations with unsupervised learning.
dataset_un = torchvision.datasets.STL10('./data', 'unlabeled', download=True, transform=unlabeled_transform)
def train_classifier(encoder, cls, dataloader, epochs=100, supervised=False):
    '''
    Args:
        encoder: trained/untrained encoder for unsupervised/supervised training.
        cls: linear classifier.
        dataloader: train partition.
        supervised:

    Return:
        cls: linear clssifier.
    '''

    optimizer = optim.Adam(cls.parameters(), lr=0.001, weight_decay=1e-4)
    if supervised:
        optimizer = optim.Adam(list(cls.parameters())+list(encoder.parameters()), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loss_traj = []
    accuracy_traj = []

    for epoch in tqdm_notebook(range(epochs)):

        loss_epoch = 0
        corrects_epoch = 0
        for x, y in dataloader:

            batch_size = x.size(0)
            x = x.float()
            ##############################################################################
            #                               YOUR CODE HERE                               #
            ##############################################################################
            # TODO: update the parameters of the classifer. If in supervised mode, the   #
            # parameter of the encoder is also updated.                                  #
            ##############################################################################
            x, y = x, y
            optimizer.zero_grad()
            outs = cls.forward(encoder.forward(x).reshape(-1, feat_dim))
            loss = criterion(outs, y)
            loss.backward()
            optimizer.step()
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################
            _, preds = torch.max(outs, 1)
            corrects_epoch += torch.sum(preds == y.data)
            loss_epoch += loss.detach()

        loss_traj.append(loss_epoch)
        epoch_acc = corrects_epoch.double() / len(dataloader.dataset)
        accuracy_traj.append(epoch_acc)

        if epoch % 10 == 0:
            print('Epoch {}, loss {:.3f}, train accuracy {}'.format(epoch, loss_epoch, epoch_acc))

    return cls, loss_traj

def test(encoder, cls, dataloader):
    '''
    Calculate the accuracy of the trained linear classifier on the test set.
    '''
    cls.eval()

    loss_epoch = 0
    corrects_epoch = 0
    for x, y in dataloader:

        x = x.float()
        batch_size = x.size(0)
        x, y = x, y
        h = encoder(x).view(batch_size, -1)
        outs = cls(h)
        _, preds = torch.max(outs, 1)
        corrects_epoch += torch.sum(preds == y.data)

    epoch_acc = corrects_epoch.double() / len(dataloader.dataset)
    print('Test accuracy {}'.format(epoch_acc))

# dataset pre-processing
df = pd.read_csv('captions.txt', delimiter='|')
img_color = cv2.imread('flickr30k/Images/'+ str(df['image_name'][0]),1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
df.to_csv("captions.csv", index=False)
df = pd.read_csv("captions.csv")
df.head()

image_path = "/flickr30k/images/flickr30k_images"
captions_path = "/flickr30k"
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
        text_loss = criterion(logits, torch.arange(batch_size).cuda())
        image_loss = criterion(logits.T, torch.arange(batch_size).cuda().T)
        loss = (text_loss + image_loss)/2
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return loss.mean()
    
def make_train_valid_dfs():
    dataframe = pd.read_csv("captions.csv")
    dataframe = dataframe[dataframe.reset_index().index % 5 == 0]
    max_id = dataframe["id"].max() + 1
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image_name"].values,
        dataframe["caption_text"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if mode == "train" else False,
        drop_last=True
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for i, batch in enumerate(tqdm_object):
        batch = {k: v for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

        if i % 100 == 0:
            print('loss:', loss.item() / count)
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

train_df, valid_df = make_train_valid_dfs()
tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
train_loader = build_loaders(train_df, tokenizer, mode="train")
valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


model = CLIPModel()
params = [
    {"params": model.image_encoder.parameters(), "lr": image_encoder_lr},
    {"params": model.text_encoder.parameters(), "lr": text_encoder_lr},
    {"params": itertools.chain(
        model.image_projection.parameters(), model.text_projection.parameters()
    ), "lr": head_lr, "weight_decay": weight_decay}
]
optimizer = torch.optim.AdamW(params, weight_decay=0.)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=patience, factor=factor
)
step = "epoch"

best_loss = float('inf')
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
    model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(model, valid_loader)

    if valid_loss.avg < best_loss:
        best_loss = valid_loss.avg
        torch.save(model.state_dict(),  "best.pt")
        print("Saved Best Model!")

    lr_scheduler.step(valid_loss.avg)