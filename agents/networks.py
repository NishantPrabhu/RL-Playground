
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class PixelEncoder(nn.Module):
    
    def __init__(self, input_ch, hidden_ch=32, out_ch=1024):
        super(PixelEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_ch, hidden_ch, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch * 2, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(hidden_ch * 2, hidden_ch * 2, kernel_size=3, stride=1, bias=False)
        self.conv4 = nn.Conv2d(hidden_ch * 2, out_ch, kernel_size=7, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.out_dim = out_ch 
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        return x
    
    
class QNetwork(nn.Module):
    
    def __init__(self, input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions):
        super(QNetwork, self).__init__()
        
        self.encoder = PixelEncoder(input_ch, enc_hidden_ch, enc_fdim)
        self.q_action = nn.Sequential(
            nn.Linear(self.encoder.out_dim, q_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_hidden_dim, n_actions)
        )
        self.init()
        
    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 1)
        
    def forward(self, obs):
        fs = self.encoder(obs)
        q_vals = self.q_action(fs)
        return q_vals, fs
    

class DuelQNetwork(nn.Module):
    
    def __init__(self, input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions):
        super(DuelQNetwork, self).__init__()
        
        self.encoder = PixelEncoder(input_ch, enc_hidden_ch, enc_fdim)
        self.q_action = nn.Sequential(
            nn.Linear(self.encoder.out_dim // 2, q_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_hidden_dim, n_actions)
        )
        self.q_value = nn.Sequential(
            nn.Linear(self.encoder.out_dim // 2, q_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_hidden_dim, 1)
        )
        self.init()
        
    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 1)
        
    def forward(self, obs):
        fs = self.encoder(obs)
        fs_action, fs_value = torch.split(fs, self.encoder.out_dim // 2, 1)
        action_q = self.q_action(fs_action)
        value_q = self.q_value(fs_value)
        q_vals = value_q + (action_q - action_q.mean(-1, keepdim=True))
        return q_vals, fs