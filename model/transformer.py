

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

def create_PositionalEncoding(input_dim, max_seq_len=2000): 
    position_encoding = np.array([ 
        [pos / np.power(10000, 2.0 * (j // 2) / input_dim) for j in range(input_dim)] 
        for pos in range(max_seq_len)]) 
    
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
    position_encoding = nn.Parameter(position_encoding, requires_grad=False) 
    
    return position_encoding

class Multihead_attention(nn.Module):

    def __init__(self, self_attn, embed_dim, num_heads, qdim=None, kdim=None, vdim=None, dropout=0., bias=True):
        super(Multihead_attention, self).__init__()
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        if self_attn:
            self.project_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        else:
            self.project_q = nn.Linear(self.qdim, embed_dim, bias=bias)
            self.project_k = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.project_v = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5
        self.self_attn = self_attn
        
    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):

        bsz, tgt_len, _ = query.size()
        if self.self_attn:
            Q, K, V = self.project_qkv(query).chunk(3, dim=-1)
        else:
            Q = self.project_q(query)
            K = self.project_k(key)
            V = self.project_v(value)
        Q = Q * self.scaling
        Q = Q.transpose(0, 1).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        K = K.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = K.size(1)
        attn_output_weights = torch.bmm(Q, K.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1) if key_padding_mask.dim() == 3 else key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask, float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, V)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        attn_output = self.project_out(attn_output)

        return attn_output

class TransformerEncoder(nn.Module):
    def __init__(self, self_attn, embed_dim, qdim=None, kdim=None, ffn_embed_dim=2304, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = self_attn
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation)

        self.attention = Multihead_attention(self_attn, embed_dim, num_heads, qdim, kdim, kdim, attention_dropout)
        self.attention_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
    
    def add_position(self, x, position=None, mask=None):

        if position is None:
            return x
        else:
            B, T = x.shape[:2]
            position = position[:T].unsqueeze(dim=0).repeat(B, 1, 1)  # -> B, T, C
            position = position*((1 - mask.unsqueeze(-1).type_as(x))) if mask is not None else position
            return x + position

    def forward(self, query, key=None, value=None, query_position=None, key_position=None, key_padding_mask=None, attn_mask=None):

        residual = query
        query = self.add_position(query, query_position)
        key = self.add_position(key, key_position) if not self.self_attn else key
            
        x = self.attention(query, key, value, key_padding_mask, attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.attention_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x

class Transformer(nn.Module):

    def __init__(self, self_attn, num_layers, embed_dim, qdim=None, kdim=None, ffn_embed_dim=2304, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = self_attn
        self.query_position = create_PositionalEncoding(embed_dim)
        self.query_input_norm = nn.LayerNorm(embed_dim)
        self.key_position = create_PositionalEncoding(kdim) if not self_attn else None
        self.key_input_norm = nn.LayerNorm(kdim) if not self_attn else None

        self.layers = nn.ModuleList([TransformerEncoder(self_attn, embed_dim, qdim, kdim, ffn_embed_dim, num_heads,
                                                        dropout, attention_dropout, activation) for _ in range(num_layers)])

        self._reset_parameters()     

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.kaiming_uniform_(p)

    def forward(self, query, key=None, key_padding_mask=None, attn_mask=None):
        output = self.query_input_norm(query)
        value = None
        if self.self_attn:
            if key is not None:
                print("you don't need to provide key input in forward function when doing self attention")
        else:
            assert key is not None, 'key input should be provided for doing cross attention.'
            key = self.key_input_norm(key)
            value = key

        for layer in self.layers:
            output = layer(output, key, value, self.query_position, self.key_position, key_padding_mask, attn_mask)

        return output

def build_transformer(self_attn, num_layers, embed_dim, qdim=None, kdim=None, ffn_embed_dim=2304, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='gelu'):

    if qdim is not None:
        assert embed_dim == qdim
    if self_attn:
        if kdim is not None:
            print("you don't need to provide kdim in build_transformer when doing self attention")
    else:
        assert kdim is not None, 'kdim should be provided for cross attention.'

    return Transformer(self_attn, num_layers, embed_dim, qdim, kdim, ffn_embed_dim, num_heads, dropout, attention_dropout, activation)

