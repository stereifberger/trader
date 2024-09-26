import torch
import torch.nn as nn

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        output, _ = self.decoder(hidden)
        output = self.fc(output)
        return output

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        output = self.fc(x)
        return output

def get_model(model_type, **kwargs):
    if model_type == 'lstm':
        return LSTMEncoderDecoder(**kwargs)
    elif model_type == 'transformer':
        return TransformerModel(**kwargs)
    else:
        raise ValueError("Unknown model type: Choose either 'lstm' or 'transformer'")
