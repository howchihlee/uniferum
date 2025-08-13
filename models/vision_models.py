from typing import Sequence
import timm_3d
import torch
from torch import nn

class EfficientNetViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = timm_3d.create_model(**config["model_args"])
        self.feature_dim = config["feature_dim"]  # checkout efficientnet specs
        self.llm_emb_dim = config["llm_emb_dim"]
        self.seqlen = config["seqlen"]
        self.emb_start = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.emb_end = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.lm_adaptor = nn.Sequential(
            #nn.TransformerEncoderLayer(
            #    d_model=self.feature_dim, nhead=config["nhead"], batch_first=True
            #),
            nn.Linear(self.feature_dim, self.llm_emb_dim),
        )

    def forward(self, imgs):
        """
        take imgs and output sequence of embeddings for llama
        imgs: (B, in_channels, H, W, D), (H, W, D) = image_sizes
        out: (B, seqlen, lm_dim)
        """
        batchsize = imgs.size(0)
        # print(imgs.shape)
        out = imgs
        out = self.model(out)
        out = out.view(batchsize, out.size(1), -1).swapaxes(1, 2)  # B, 8x8x8 , 1280
        out = self.lm_adaptor(out)  # B, 512, 1280
        
        emb_s = self.emb_start.expand(batchsize, -1, -1)
        emb_e = self.emb_end.expand(batchsize, -1, -1)
        out = torch.cat([emb_s, out, emb_e], 1)
        # print(out.shape)
        return out
