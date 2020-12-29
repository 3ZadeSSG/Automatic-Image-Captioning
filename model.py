import torch
import torch.nn as nn
import torchvision.models as models


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
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            
        self.hidden = (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))
        
        
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        output = self.fc1(lstm_out)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        batch_size = inputs.shape[0]
        
        #print("Batch Size: ",batch_size)
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            
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
    
    
def save_checkpoint(model,fileLocation):
    torch.save(model.state_dict(),fileLocation)
      
def load_checkpoint(model,fileLocation):
    return model.load_state_dict(torch.load(fileLocation))


        
        
        
        
        
        
        
        
        
        