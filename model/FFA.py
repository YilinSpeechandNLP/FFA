# 2023年月01日16时18分39秒

import numpy as np
import torch
import torch.nn as nn
from model.transformer import build_transformer
from model.FFM_block import build_FFM_block
from model.Co import build_Co
from transformers import BertModel
# import pytorch_revgrad





class FFA(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # feature extraction
        self.audio_self_Trans = build_transformer(self_attn=True, num_layers=num_layers[0], embed_dim=input_dim[0],
                                                  ffn_embed_dim=ffn_embed_dim[0], num_heads=num_heads,
                                                  )
        self.text_self_Trans = build_transformer(self_attn=True, num_layers=num_layers[0], embed_dim=input_dim[1],
                                                 ffn_embed_dim=ffn_embed_dim[1], num_heads=num_heads,
                                                 )
        # modality interaction
        self.at_cross_Trans = build_Co(num_layers=num_layers[1], embed_dim=input_dim[0], kdim=input_dim[1],
                                         ffn_embed_dim=ffn_embed_dim[0], num_heads=num_heads)
        self.ta_cross_Trans = build_Co(num_layers=num_layers[1], embed_dim=input_dim[1], kdim=input_dim[0],
                                         ffn_embed_dim=ffn_embed_dim[1], num_heads=num_heads)
        # Deep fusion
        self.last_audio_self_Trans = build_FFM_block(self_attn=True, num_layers=num_layers[2],
                                                                embed_dim=input_dim[0], ffn_embed_dim=ffn_embed_dim[0],
                                                                num_heads=num_heads,
                                                                )
        self.last_text_self_Trans = build_FFM_block(self_attn=True, num_layers=num_layers[2],
                                                               embed_dim=input_dim[1], ffn_embed_dim=ffn_embed_dim[1],
                                                               num_heads=num_heads,
                                                               )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        fc_dim = self.input_dim[0] + self.input_dim[1]
        self.classifier = nn.Sequential(
            # pytorch_revgrad.RevGrad(),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim // 2, fc_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim // 4, num_classes),
        )
        self.classifier1 = nn.Sequential(

            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim // 2, fc_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim // 4, 4),
        )
        self.classifier2 = nn.Sequential(

            nn.Linear(fc_dim, fc_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim // 2, fc_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim // 4, num_classes),
        )

    def forward(self, x_a=None, x_t=None, x_a_padding_mask=None, x_t_padding_mask=None,
                x1_a=None, x1_t=None, x1_a_padding_mask=None, x1_t_padding_mask=None,
                x2_a: torch.Tensor = None, x2_t: torch.Tensor = None, x2_a_padding_mask=None, x2_t_padding_mask=None):
        x, x1, x2 = None, None, None

        if x_a is not None:
            x_t=self.outputs = self.bert(
                x_t,
                attention_mask=x_t_padding_mask,
            )
            x_t=x_t[0]
            x_t_padding_mask = x_t_padding_mask == 0
            # x_t_padding_mask=x_t_padding_mask.bool()
            x_a = self.audio_self_Trans(x_a, key_padding_mask=x_a_padding_mask)
            x_t = self.text_self_Trans(x_t, key_padding_mask=x_t_padding_mask)

            x_at = x_a
            x_ta = x_t
            x_at = self.at_cross_Trans(x=x_at, k=x_t, key_padding_mask=x_t_padding_mask)
            x_ta = self.ta_cross_Trans(x=x_ta, k=x_a, key_padding_mask=x_a_padding_mask)

            x_a = x_a + x_at
            x_t = x_t + x_ta

            x_a = self.last_audio_self_Trans(x_a, key_padding_mask=None).transpose(1, 2)
            x_t = self.last_text_self_Trans(x_t, key_padding_mask=None).transpose(1, 2)

            x_a = self.avgpool(x_a).view(x_a.shape[0], -1)
            x_t = self.avgpool(x_t).view(x_t.shape[0], -1)

            x = torch.cat((x_a, x_t), dim=-1)
            x = self.classifier(x)
        if x1_a is not None:
            x1_t = self.outputs = self.bert(
                x1_t,
                attention_mask=x1_t_padding_mask,
            )
            x1_t = x1_t[0]
            # x1_t_padding_mask = x1_t_padding_mask.bool()
            x1_t_padding_mask = x1_t_padding_mask == 0
            x1_a = self.audio_self_Trans(x1_a, key_padding_mask=x1_a_padding_mask)
            x1_t = self.text_self_Trans(x1_t, key_padding_mask=x1_t_padding_mask)

            x1_at = x1_a
            x1_ta = x1_t
            x1_at = self.at_cross_Trans(x=x1_at, k=x1_t, key_padding_mask=x1_t_padding_mask)
            x1_ta = self.ta_cross_Trans(x=x1_ta, k=x1_a, key_padding_mask=x1_a_padding_mask)

            x1_a = x1_a + x1_at
            x1_t = x1_t + x1_ta

            x1_a = self.last_audio_self_Trans(x1_a, key_padding_mask=None).transpose(1, 2)
            x1_t = self.last_text_self_Trans(x1_t, key_padding_mask=None).transpose(1, 2)

            x1_a = self.avgpool(x1_a).view(x1_a.shape[0], -1)
            x1_t = self.avgpool(x1_t).view(x1_t.shape[0], -1)

            x1 = torch.cat((x1_a, x1_t), dim=-1)
            x1 = self.classifier1(x1)
        if x2_a is not None:
            x2_t = self.outputs = self.bert(
                x2_t,
                attention_mask=x2_t_padding_mask,
            )
            x2_t = x2_t[0]
            # x2_t_padding_mask = x2_t_padding_mask.bool()
            x2_t_padding_mask = x2_t_padding_mask == 0
            x2_a = self.audio_self_Trans(x2_a, key_padding_mask=x2_a_padding_mask)
            x2_t = self.text_self_Trans(x2_t, key_padding_mask=x2_t_padding_mask)

            x2_at = x2_a
            x2_ta = x2_t
            x2_at = self.at_cross_Trans(x=x2_at, k=x2_t, key_padding_mask=x2_t_padding_mask)
            x2_ta = self.ta_cross_Trans(x=x2_ta, k=x2_a, key_padding_mask=x2_a_padding_mask)

            x2_a = x2_a + x2_at
            x2_t = x2_t + x2_ta

            x2_a = self.last_audio_self_Trans(x2_a, key_padding_mask=None).transpose(1, 2)
            x2_t = self.last_text_self_Trans(x2_t, key_padding_mask=None).transpose(1, 2)

            x2_a = self.avgpool(x2_a).view(x2_a.shape[0], -1)
            x2_t = self.avgpool(x2_t).view(x2_t.shape[0], -1)

            x2 = torch.cat((x2_a, x2_t), dim=-1)
            x2 = self.classifier2(x2)

        return x, x1, x2


def build_FFA(**kwargs):
    return FFA(**kwargs)


