
import torch.nn as nn

from model.FFM_block import build_FFM_block

class Co(nn.Module):
    def __init__(self, embed_dim, kdim, ffn_embed_dim, num_heads):
        super().__init__()
        self.FFM_1 = build_FFM_block(self_attn=False, num_layers=1, embed_dim=embed_dim, kdim=kdim, ffn_embed_dim=ffn_embed_dim, num_heads=num_heads)
        self.FFM_2 = build_FFM_block(self_attn=True, num_layers=1, embed_dim=embed_dim, ffn_embed_dim=ffn_embed_dim, num_heads=num_heads)
    
    def forward(self, x, k, key_padding_mask):
        x = self.FFM_1(query=x, key=k, key_padding_mask=key_padding_mask)
        x = self.FFM_2(query=x, key_padding_mask=None)
        return x
    
class Cos(nn.Module):
    def __init__(self, embed_dim, kdim, ffn_embed_dim, num_layers, num_heads):
        super().__init__()
        self.Cos_model = nn.ModuleList([Co(embed_dim, kdim, ffn_embed_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, k, key_padding_mask):
        residual = x
        for layer in self.Cos_model:
            x = layer(x, k, key_padding_mask)
        x = x + residual
        return x

def build_Co(embed_dim, kdim, ffn_embed_dim, num_layers, num_heads):
    '''
    forward: x, k, key_padding_mask
    '''
    return Cos(embed_dim, kdim, ffn_embed_dim, num_layers, num_heads)

