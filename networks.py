
import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F


class Attention(nn.Module):
    
    def __init__(self, model_dim, num_heads):
        super(Attention, self).__init__()
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.nheads = num_heads
        self.mdim = model_dim
        
    def forward(self, x):
        bs, seqlen, _ = x.size()
        x_norm = self.norm(x)
        
        q = self.query(x_norm).view(bs, seqlen, self.nheads, -1).permute(0, 2, 1, 3).contiguous()
        k = self.key(x_norm).view(bs, seqlen, self.nheads, -1).permute(0, 2, 1, 3).contiguous()
        v = self.value(x_norm).view(bs, seqlen, self.nheads, -1).permute(0, 2, 1, 3).contiguous()
        
        scores = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(self.mdim)
        probs = F.softmax(scores, -1)
        out = torch.einsum('bhij,bhjd->bhid', probs, v).permute(0, 2, 1, 3).contiguous().view(bs, seqlen, -1)
        out = out + x 
        return out, probs 
    
    
class Feedforward(nn.Module):
    
    def __init__(self, model_dim):
        super(Feedforward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim)
        )
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        x_norm = self.norm(x)
        out = self.mlp(x_norm) + x
        return out
    
    
class Embedding(nn.Module):
    
    def __init__(self, n_embeds, embed_dim, add_action_embeds=True, n_actions=None):
        super(Embedding, self).__init__()
        self.add_action_embeds = add_action_embeds
        self.n_embeds = n_embeds
        self.n_actions = n_actions
        
        self.pos_embeds = nn.Embedding(n_embeds+1, embed_dim)
        if self.add_action_embeds:
            assert isinstance(n_actions, int), 'Number of actions should be integer for adding action embeds'
            self.action_embeds = nn.Embedding(n_actions, embed_dim)
            
    def forward(self, x):
        bs, n, c = x.size() 
        
        # Position embeddings
        pos_embed = self.pos_embeds(torch.arange(n+1).to(x.device))
        pos_embed = pos_embed.unsqueeze(0).repeat(bs, 1, 1)
        x = torch.cat([torch.zeros(bs, 1, c).to(x.device), x], 1) + pos_embed
        
        # Action embeddings
        if self.add_action_embeds:
            act_embed = self.action_embeds(torch.arange(self.n_actions).to(x.device))
            act_embed = act_embed.unsqueeze(0).repeat(bs, 1, 1)
            x = torch.cat([x, act_embed], 1)
            
        return x
        
    
class Encoder(nn.Module):
    
    def __init__(self, input_shape, n_layers, n_heads, model_dim, patch_size, n_embeds, 
                 add_action_embeds=False, n_actions=None):
        super(Encoder, self).__init__()
        h, w, c = input_shape
        n_patches = (h // patch_size) * (w // patch_size)
        self.n_layers = n_layers
        
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.upscale = nn.Linear(patch_size * patch_size * c, model_dim)
        self.embedding = Embedding(n_embeds, model_dim, add_action_embeds, n_actions)
        self.attn_layers = nn.ModuleList([Attention(model_dim, n_heads) for _ in range(n_layers)])
        self.feedfwd_layers = nn.ModuleList([Feedforward(model_dim) for _ in range(n_layers)])
        
    def forward(self, inp):
        x = self.unfold(inp).permute(0, 2, 1).contiguous()
        x = self.embedding(self.upscale(x))
        attn_probs = {}
        
        for i in range(self.n_layers):
            x, attn = self.attn_layers[i](x)
            x = self.feedfwd_layers[i](x)
            attn_probs[i] = attn.detach().cpu()
        
        return x, attn_probs


class ConvBaseEncoder(nn.Module):

    def __init__(self, input_shape, n_layers, n_heads, model_dim, patch_size, n_embeds,
                 add_action_embeds=False, n_actions=None):
        super(ConvBaseEncoder, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_shape[-1], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        with torch.no_grad():
            x = torch.randn(1, input_shape[-1], *input_shape[:2])
            out = self.conv_base(x)
            _, c, h, w = out.size()
            
        n_patches = (h // patch_size) * (w // patch_size)
        self.n_layers = n_layers 
        
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.upscale = nn.Linear(patch_size * patch_size * c, model_dim)
        self.embedding = Embedding(n_embeds, model_dim, add_action_embeds, n_actions)
        self.attn_layers = nn.ModuleList([Attention(model_dim, n_heads) for _ in range(n_layers)])
        self.feedfwd_layers = nn.ModuleList([Feedforward(model_dim) for _ in range(n_layers)])

    def forward(self, inp):
        x = self.conv_base(inp)
        x = self.unfold(x).permute(0, 2, 1).contiguous()
        x = self.embedding(self.upscale(x))
        attn_probs = {}
        
        for i in range(self.n_layers):
            x, attn = self.attn_layers[i](x)
            x = self.feedfwd_layers[i](x)
            attn_probs[i] = attn.detach().cpu()
        
        return x, attn_probs

    
class QNetwork(nn.Module):
    
    def __init__(self, model_dim, hidden_dim, n_actions, cls_in_qfunc=False):
        super(QNetwork, self).__init__()
        self.n_actions = n_actions
        self.cls_in_qfunc = cls_in_qfunc                
        self.q_action = nn.Sequential(
            nn.Linear(2 * model_dim if cls_in_qfunc else model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        cls_embed, action_embeds = x[:, 0, :], x[:, -self.n_actions:, :]
        if self.cls_in_qfunc:
            inp = torch.cat([action_embeds, cls_embed.unsqueeze(1).repeat(1, self.n_actions, 1)], 1)
        else:
            inp = action_embeds 
            
        q_vals = self.q_action(inp).squeeze(-1)
        return q_vals 
    
    
class DuelQNetwork(nn.Module):
    
    def __init__(self, model_dim, hidden_dim, n_actions, cls_in_qfunc=False):
        super(DuelQNetwork, self).__init__()
        self.n_actions = n_actions 
        self.cls_in_qfunc = cls_in_qfunc
        self.q_action = nn.Sequential(
            nn.Linear(2 * model_dim if cls_in_qfunc else model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q_value = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        cls_embed, action_embeds = x[:, 0, :], x[:, -self.n_actions:, :]
        if self.cls_in_qfunc:
            inp = torch.cat([action_embeds, cls_embed.unsqueeze(1).repeat(1, self.n_actions, 1)], 1)
        else:
            inp = action_embeds 
            
        act_val = self.q_action(inp).squeeze(-1)
        state_val = self.q_value(cls_embed)
        q_vals = state_val + (act_val - act_val.mean(-1, keepdim=True))
        return q_vals