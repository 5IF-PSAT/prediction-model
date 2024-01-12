from torch import nn
import torch
from torch.nn import functional


class Mixed_NN(nn.Module):
    def __init__(self):
        super(Mixed_NN, self).__init__()

        #this is for mobile use adaptation
        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()

        self.layer1 = nn.Sequential(
            nn.Dropout(p = 0.25),
            nn.Conv1d(9, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )#output size = 6


        self.layer2 = nn.Sequential(
            nn.Linear(6*16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,8)
        )

        self.numeric_features_ = nn.Sequential(
            nn.Linear(1,4),
            nn.ReLU(inplace=True),
            nn.Linear(4,8)
        )

        self.combined_features_ = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(16,1)
        )


    def forward(self, x, y):
         
        out_ts = self.layer1(x)
        out_ts = out_ts.view(out_ts.size(0), -1)
        out_ts = self.layer2(out_ts)

        out_num = self.numeric_features_(y)

        out = torch.cat((out_ts,out_num) , dim =1)

        out = self.combined_features_(out)
        
        return out