import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # print(input_seq.shape)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # print(input_seq.size())
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred1, pred2, pred3,pred4 = self.fc1(output), self.fc2(output), self.fc3(output),self.fc4(output)
        pred1, pred2, pred3,pred4 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :],pred4[:, -1, :]

        # pred = torch.cat([pred1, pred2], dim=0)
        pred = torch.stack([pred1, pred2, pred3,pred4], dim=0).reshape(batch_size,4,-1)

        return pred