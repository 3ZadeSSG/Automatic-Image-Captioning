"""
IMPORTANT : I have not written any exception handling here since this is just a prototype module.
            But exceptions can be logged into database with "traceback.format_exec()"
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from vocabulary import Vocabulary


# This python script will be called by Node.js as child process into project root directory, hence file paths will be from one level up.

ENCODER_CNN_CHECKPOINT = "python_models/saved_models/encoderEpoch_2.pth"
DECODER_LSTM_RNN_CHECKPOINT = "python_models/saved_models/decoderEpoch_2.pth"
VOCAB_FILE = "python_models/saved_models/vocab.pkl"

# Globally Set device to CPU since GPU won't be necessary for making just one prediction at a time
device = "cpu"


#=========================================================================
# Encoder - Decoder Model Class to be used for Generating Caption
#=========================================================================

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(num_embeddings = self.vocab_size,
                                  embedding_dim = self.embed_size)
        self.lstm = nn.LSTM(input_size = self.embed_size,
                           hidden_size = self.hidden_size,
                           num_layers = self.num_layers,
                           batch_first = True)
        self.fc1 = nn.Linear(in_features = self.hidden_size,
                            out_features = self.vocab_size)


    def forward(self, features, captions):
        captions = captions[:,:-1]
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings),dim=1)
        batch_size = features.shape[0]

        self.hidden = (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        output = self.fc1(lstm_out)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        output = []
        batch_size = inputs.shape[0]

        self.hidden = (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))

        while True:
            lstm_out, self.hidden = self.lstm(inputs, self.hidden)
            linear_out = self.fc1(lstm_out.squeeze(1))
            top_score = linear_out.max(1)[1]
            output.append(top_score.cpu().numpy()[0].item())
            if top_score == 1:
                break
            inputs = self.embed(top_score)
            inputs = inputs.unsqueeze(1)
        return output





def clean_sentence(output,vocab):
    cleaned = []
    for index in output:
        cleaned.append(vocab.idx2word[index])
    cleaned = cleaned[1:-1]
    sentence = ' '.join(cleaned)
    return sentence

def generateCaptions(image_path):
    vocab = Vocabulary(vocab_file=VOCAB_FILE,vocab_from_file=True,vocab_threshold=None)

    # Model parameters
    embed_size = 512
    hidden_size = 512
    vocab_size = len(vocab)

    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=2)

    # Load the trained weights.
    encoder.load_state_dict(torch.load(ENCODER_CNN_CHECKPOINT))
    decoder.load_state_dict(torch.load(DECODER_LSTM_RNN_CHECKPOINT))

    # Put model to evaluation mode
    encoder.eval()
    decoder.eval()

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)

    transform_predict = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


    image = Image.open(image_path).convert('RGB')    # Convert to RGB in case additonal channel is there in image
    original_image = np.copy(image)
    image = torch.unsqueeze(transform_predict(image), 0).to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output, vocab)
    return {"sentence":sentence}





