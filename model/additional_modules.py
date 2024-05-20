import torch
import torch.nn as nn
import math

class Gate_Attention(nn.Module):
    def __init__(self, num_hidden_a, num_hidden_b, num_hidden):
        super(Gate_Attention, self).__init__()
        self.hidden = num_hidden
        self.w1 = nn.Parameter(torch.Tensor(num_hidden_a, num_hidden))
        self.w2 = nn.Parameter(torch.Tensor(num_hidden_b, num_hidden))
        self.bias = nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()

    def reset_parameter(self):
        stdv1 = 1. / math.sqrt(self.hidden)
        stdv2 = 1. / math.sqrt(self.hidden)
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, a, b):
        wa = torch.matmul(a, self.w1)
        wb = torch.matmul(b, self.w2)
        gated = wa + wb + self.bias
        gate = torch.sigmoid(gated)
        output = gate * a + (1 - gate) * b
        return output  # Clone the tensor to make it out of place operation
    
class LSTM_fc(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_seq_len, output_size):
        super(LSTM_fc, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=2*hidden_size, out_features=output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Select the output from the last time step
        # output = out[:, -1, :].clone()  # Clone the tensor to make it out of place operation
        # print(out[:, -1, :].shape)
        
        output = self.fc(out[:, :self.output_seq_len, :])

        return output
    
    
class Conv1d_fc(nn.Module):
    def __init__(self, encoder_embed_dim , llm_embed_dim, kernel_size, stride, padding):
        super(Conv1d_fc, self).__init__()
        
        #input_channels, output_channels - embedding dim of encoder
        # conv on seq_len
        
        self.conv1d = nn.Conv1d(in_channels=encoder_embed_dim, out_channels=encoder_embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.fc = nn.Linear(in_features=encoder_embed_dim, out_features=llm_embed_dim)

    def forward(self, x):
        
        x = self.conv1d(x.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        output = self.fc(x)
        
        return output

    
class FC_head(nn.Module):
    def __init__(self, num_classes, hidden_dim, llm_embed_dim, add_pooling = False):
        super(FC_head, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.llm_embed_dim = llm_embed_dim
        self.add_pooling = add_pooling
        
        self.pooling = nn.Linear(in_features=llm_embed_dim, out_features=llm_embed_dim)
        self.activation = nn.Tanh()
        
        self.fc1 = nn.Linear(in_features=llm_embed_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        
    def forward(self, x):
        if self.add_pooling:
            x = self.pooling(x)
            x = self.activation(x)
        
        x = torch.mean(x, dim=1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
        
        
        
        
        
        
    
