import torch
import torch.nn as nn
from einops import repeat

from models.aggregators.aggregator import BaseAggregator
from models.aggregators.model_utils import Attention, FeedForward, PreNorm


class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x, register_hook=register_hook) + x
            x = ff(x) + x
        return x


class Transformer(BaseAggregator):
    def __init__(
            self,
            *,
            num_classes,
            input_dim=2048,
            input_dim1=768,
            dim=512,
            depth=2,
            heads=8,
            mlp_dim=512,
            pool='cls',
            dim_head=64,
            dropout=0.2,
            emb_dropout=0.,
            pos_enc=None,
    ):
        super(BaseAggregator, self).__init__()
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean (mean pooling)'

        self.projection = nn.Sequential(nn.Linear(input_dim, heads * dim_head, bias=True), nn.ReLU())
        self.projection1 = nn.Sequential(nn.Linear(input_dim1, heads * dim_head, bias=True), nn.ReLU())
        self.mlp_head = nn.Sequential(nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, num_classes))
        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.pos_enc = pos_enc

    def forward(self, x, x1, coords=None, register_hook=False):
        b, _,_ = x.shape

        x = self.projection(x)

        if self.pos_enc:
            x = x + self.pos_enc(coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, register_hook=register_hook)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.norm(x)
        x = self.mlp_head(x)

        # b, _, _ = x1.shape
        #
        # x1 = self.projection1(x1)
        #
        # if self.pos_enc:
        #     x1 = x1 + self.pos_enc(coords)
        #
        # if self.pool == 'cls':
        #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        #     x1 = torch.cat((cls_tokens, x1), dim=1)
        #
        # x1 = self.dropout(x1)
        # x1 = self.transformer(x1, register_hook=register_hook)
        # x1 = x1.mean(dim=1) if self.pool == 'mean' else x1[:, 0]
        # x1 = self.norm(x1)
        # x1 = self.mlp_head(x1)

        # x = torch.squeeze(x)
        # x1 = torch.squeeze(x1)
        #
        # ##合并特征
        # concatenated_feature = torch.cat((x, x1), dim=0).unsqueeze(0)
        # out = torch.mean(concatenated_feature)  # nn.Sigmoid()(torch.mean(concatenated_feature))
        # out = torch.unsqueeze(out, 0).unsqueeze(1)
        out = x#(x + x1)

        return out

# transformer = Transformer(num_classes=2)
# transformer(torch.rand(1, 1, 2048))

