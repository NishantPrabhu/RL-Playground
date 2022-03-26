
import math
import torch 
import random 
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
    
    
# ====================================================================
# Layers and heads
# ====================================================================
    
class MultiheadSelfAttention(nn.Module):
    
    def __init__(self, n_heads, model_dim):
        super(MultiheadSelfAttention, self).__init__()
        model_dim = n_heads * (model_dim // n_heads)                    # Adjust model_dim to be divisible by n_heads
        self.head_dim = (model_dim // n_heads)
        self.n_heads = n_heads
        
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.layernorm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        bs, n, c = x.size()
        x_norm = self.layernorm(x)
        
        q = self.query(x_norm).view(bs, n, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.key(x_norm).view(bs, n, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.value(x_norm).view(bs, n, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        attn_scores = torch.einsum('bhid,bhjd->bhij', [q, k]) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, -1)
        attn_probs = self.dropout(attn_probs)
        
        out = torch.einsum('bhij,bhjd->bhid', [attn_probs, v])
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, n, -1) + x
        return out, attn_probs
    

class Feedforward(nn.Module):
    
    def __init__(self, model_dim, hidden_dim):
        super(Feedforward, self).__init__()
        self.layernorm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim))
        
    def forward(self, x):
        x_norm = self.layernorm(x)
        out = self.head(x_norm)
        out = out + x 
        return out

    
class ClusteringHead(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_clusters):
        super(ClusteringHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_clusters))
    
    def forward(self, x):
        logits = self.head(x)
        probs = F.softmax(logits, -1)
        return probs
    
    
# ====================================================================
# Encoders
# ====================================================================
        
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
        fs = self.relu(self.conv3(x))
        pooled_fs = self.relu(self.conv4(fs))
        pooled_fs = torch.flatten(pooled_fs, 1)
        return pooled_fs, fs
    

class ViTLayer(nn.Module):
    
    def __init__(self, n_heads, model_dim, feedfwd_dim):
        super(ViTLayer, self).__init__()
        self.attention = MultiheadSelfAttention(n_heads, model_dim)
        self.feedfwd = Feedforward(model_dim, feedfwd_dim)        
        
    def forward(self, x):
        x, attn_probs = self.attention(x)
        x = self.feedfwd(x)
        return x, attn_probs
        
    
class ViTEncoder(nn.Module):
    
    def __init__(self, n_layers, n_heads, n_patches, model_dim, img_input_dim, action_input_dim):
        super(ViTEncoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_patches = n_patches
        self.img_input_dim = img_input_dim
        self.action_input_dim = action_input_dim
        self.model_dim = (model_dim // self.n_heads) * self.n_heads 

        self.img_upscale = nn.Linear(self.img_input_dim, self.model_dim, bias=False)
        self.action_upscale = nn.Linear(self.action_input_dim, self.model_dim, bias=False)
        self.pos_embeds = nn.Embedding(self.n_patches+1, self.model_dim)
        self.enc_layers = nn.ModuleList([
            ViTLayer(self.n_heads, self.model_dim, self.model_dim * 2) for _ in range(self.n_layers)
        ])
        
    def add_embeddings(self, x, action_embeds):
        zeros = torch.zeros(x.size(0), 1, x.size(2)).float().to(x.device)
        pos_embed = self.pos_embeds(torch.arange(self.n_patches+1).to(x.device)).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat([zeros, x], 1) + pos_embed
        x = torch.cat([x, action_embeds], 1)
        return x
        
    def forward(self, x, action_embeds):
        x = self.img_upscale(x)
        action_embeds = self.action_upscale(action_embeds)
        x = self.add_embeddings(x, action_embeds)
        
        layerwise_attn_probs = {}
        for i in range(self.n_layers):
            x, attn_probs = self.enc_layers[i](x)
            layerwise_attn_probs[f'layer{i}'] = attn_probs.detach()
        
        return x, layerwise_attn_probs 


# ====================================================================
# Q Networks
# ====================================================================
    
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
        
    def forward(self, obs, replay=False):
        pooled_fs, fs = self.encoder(obs)
        
        if replay and random.uniform(0, 1) < 0.5:
            noise = torch.from_numpy(np.random.normal(0, 0.01, size=pooled_fs.shape)).float().to(pooled_fs.device)
            pooled_fs += noise
            
        q_vals = self.q_action(pooled_fs)
        return q_vals, pooled_fs, fs
    

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
        
    def forward(self, obs, replay=False):
        pooled_fs, fs = self.encoder(obs)
        
        if replay and random.uniform(0, 1) < 0.5:
            noise = torch.from_numpy(np.random.normal(0, 0.01, size=pooled_fs.shape)).float().to(pooled_fs.device)
            pooled_fs += noise
        
        fs_action, fs_value = torch.split(pooled_fs, self.encoder.out_dim // 2, 1)
        action_q = self.q_action(fs_action)
        value_q = self.q_value(fs_value)
        q_vals = value_q + (action_q - action_q.mean(-1, keepdim=True))
        return q_vals, pooled_fs, fs
    
    
class VectorizedActionQNetwork(nn.Module):
    
    def __init__(self, input_ch, enc_hidden_ch, enc_fdim, n_actions):
        super(VectorizedActionQNetwork, self).__init__()
        
        self.encoder = PixelEncoder(input_ch, enc_hidden_ch, enc_fdim)
        self.action_embeds = nn.Embedding(n_actions, enc_fdim)
        self.n_actions = n_actions
        
    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 1)
        
    def forward(self, obs):
        pooled_fs, fs = self.encoder(obs)
        a_embeds = self.action_embeds(torch.arange(self.n_actions).to(obs.device))
        q_vals = torch.mm(pooled_fs, a_embeds.t())
        return q_vals, pooled_fs, fs
    
    
class AttentionQNetwork(nn.Module):
    
    def __init__(self, input_ch, enc_hidden_ch, enc_fdim, n_actions, input_res,
                 n_attn_heads, n_vit_layers, vit_fdim, action_fdim):
        super(AttentionQNetwork, self).__init__()
        
        self.encoder = PixelEncoder(input_ch, enc_hidden_ch, enc_fdim)
        with torch.no_grad():
            x = torch.randn(1, input_ch, *input_res)
            out = self.encoder.conv1(x)
            enc_out_dim, out_h, out_w = out.shape[1:]
            
        self.vit = ViTEncoder(n_vit_layers, n_attn_heads, (out_h * out_w), vit_fdim, enc_out_dim, action_fdim)
        self.action_embeds = nn.Embedding(n_actions, action_fdim)
        self.q_action = nn.Linear(self.vit.model_dim, 1)
        self.n_actions = n_actions 
        
    def load_encoder(self, src_model):
        self.encoder.load_state_dict(src_model.encoder.state_dict())
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        
    def forward(self, img):
        x = F.relu(self.encoder.conv1(img))
        x = torch.flatten(x, 2).transpose(1, 2).contiguous()
        action_emb = self.action_embeds(torch.arange(self.n_actions).to(img.device))
        action_emb = action_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        x, attn_probs = self.vit(x, action_emb)
        
        x_action = x[:, -self.n_actions:, :]
        pooled_fs = x[:, 0, :]
        q_vals = self.q_action(x_action).squeeze(-1)
        
        return q_vals, pooled_fs, x, attn_probs
    
    
# ==============================================================
#  Generative
# ==============================================================

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )
    
def upsample_block(channels, scale_factor):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
        nn.Conv2d(channels, channels, kernel_size=1, stride=1)
    )

class Generator(nn.Module):
    
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.transform = nn.Linear(input_dim, 256 * 7 * 7, bias=False)  
        self.layer1 = self._make_layer(scale_factor=3, in_planes=256, out_planes=128)
        self.layer2 = self._make_layer(scale_factor=2, in_planes=128, out_planes=64)
        self.layer3 = self._make_layer(scale_factor=2, in_planes=64, out_planes=32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        
    def _make_layer(self, scale_factor, in_planes, out_planes):
        return nn.Sequential(
            upsample_block(in_planes, scale_factor),
            conv_block(in_planes, in_planes),
            conv_block(in_planes, out_planes))
        
    def forward(self, x):
        x = self.transform(x).view(-1, 256, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.out_conv(x)
        x = torch.sigmoid(x)
        return x