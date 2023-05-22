import torch.nn as nn
import torch

class GLSTM(nn.Module):
    def __init__(self, in_features=None, out_features=None, mid_features=None, hidden_size=128*2, groups=2):
        super(GLSTM, self).__init__()
   
        hidden_size_t = hidden_size // groups
     
        self.lstm_list1 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
     
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
     
        self.groups = groups
        self.mid_features = mid_features
     
    def forward(self, x):
        out = x
        
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
        
        param1 = torch.chunk(out, self.groups, dim=-1)
        lstm1_front, (hn1_1, cn1_1)  = self.lstm_list1[0](param1[0])
        lstm1_back,  (hn1_2, cn1_2)  = self.lstm_list1[1](param1[1])
        
        out = torch.stack([lstm1_front, lstm1_back], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)
        param2 = torch.chunk(out, self.groups, dim=-1)
        
        lstm2_front, (hn2_1, cn2_1)  = self.lstm_list2[0](param2[0])
        lstm2_back,  (hn2_2, cn2_2)  = self.lstm_list2[1](param2[1])
        out = torch.cat([lstm2_front, lstm2_back], dim=-1)
        out = self.ln2(out)
        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()
        
        return out
     
class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=(0,0)):
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=output_padding)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=output_padding)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out

class CRNN(nn.Module):
    
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.conv1 = GluConv2d(in_channels=2, out_channels=16, kernel_size=(1,3), stride=(1,2))
        self.conv2 = GluConv2d(in_channels=16, out_channels=32, kernel_size=(1,3), stride=(1,2))
        self.conv3 = GluConv2d(in_channels=32, out_channels=64, kernel_size=(1,3), stride=(1,2))
        self.conv4 = GluConv2d(in_channels=64, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv5 = GluConv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=(1,2))
        
        self.glstm = GLSTM(groups=2)

        self.conv5_t_1 = GluConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv4_t_1 = GluConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1,3), stride=(1,2))
        self.conv3_t_1 = GluConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1,3), stride=(1,2))
        self.conv2_t_1 = GluConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1,3), stride=(1,2), output_padding=(0,1))
        self.conv1_t_1 = GluConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1,3), stride=(1,2))
        
        self.conv5_t_2 = GluConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv4_t_2 = GluConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1,3), stride=(1,2))
        self.conv3_t_2 = GluConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1,3), stride=(1,2))
        self.conv2_t_2 = GluConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1,3), stride=(1,2), output_padding=(0,1))
        self.conv1_t_2 = GluConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1,3), stride=(1,2))
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
                   
        self.bn5_t_1 = nn.BatchNorm2d(128)
        self.bn4_t_1 = nn.BatchNorm2d(64)
        self.bn3_t_1 = nn.BatchNorm2d(32)
        self.bn2_t_1 = nn.BatchNorm2d(16)
        self.bn1_t_1 = nn.BatchNorm2d(1)

        self.bn5_t_2 = nn.BatchNorm2d(128)
        self.bn4_t_2 = nn.BatchNorm2d(64)
        self.bn3_t_2 = nn.BatchNorm2d(32)
        self.bn2_t_2 = nn.BatchNorm2d(16)
        self.bn1_t_2 = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)
        
        self.fc1 = nn.Linear(in_features=65, out_features=65)
        self.fc2 = nn.Linear(in_features=65, out_features=65)
   
    def forward(self, x):
        out = x
        e1 = self.elu(self.bn1(self.conv1(out)))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        e3 = self.elu(self.bn3(self.conv3(e2)))
        e4 = self.elu(self.bn4(self.conv4(e3)))
        e5 = self.elu(self.bn5(self.conv5(e4)))
        out = e5
        
        out= self.glstm(out)
        out = torch.cat((out, e5), dim=1)
        
        out1 = self.elu(torch.cat((self.bn5_t_1(self.conv5_t_1(out)), e4), dim=1))
        out1 = self.elu(torch.cat((self.bn4_t_1(self.conv4_t_1(out1)), e3), dim=1))
        out1 = self.elu(torch.cat((self.bn3_t_1(self.conv3_t_1(out1)), e2), dim=1))
        out1 = self.elu(torch.cat((self.bn2_t_1(self.conv2_t_1(out1)), e1), dim=1))
        out1 = self.elu(self.bn1_t_1(self.conv1_t_1(out1)))
        
        out2 = self.elu(torch.cat((self.bn5_t_2(self.conv5_t_2(out)), e4), dim=1))
        out2 = self.elu(torch.cat((self.bn4_t_2(self.conv4_t_2(out2)), e3), dim=1))
        out2 = self.elu(torch.cat((self.bn3_t_2(self.conv3_t_2(out2)), e2), dim=1))
        out2 = self.elu(torch.cat((self.bn2_t_2(self.conv2_t_2(out2)), e1), dim=1))
        out2 = self.elu(self.bn1_t_2(self.conv1_t_2(out2)))
        
        out1 = self.fc1(out1)
        out2 = self.fc2(out2)
        
        out = torch.complex(out1, out2)
        
        return out
