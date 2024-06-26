import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
class MLP(nn.Cell):
    def __init__(self,gene_emb,cfg,shard=None):
        super().__init__()
        # const
        self.depth=cfg.enc_nlayers
        self.gene_emb=ms.Parameter(ms.Tensor(gene_emb))
        self.gene_emb.requires_grad=False
        self.mlp=nn.SequentialCell(
            nn.Dense(cfg.enc_dims,cfg.enc_dims//2,has_bias=False),
            nn.Dropout(p=0.15),
            nn.SiLU(),
            nn.Dense(cfg.enc_dims//2,cfg.enc_dims//4,has_bias=False),
            nn.Dropout(p=0.15),
            nn.SiLU(),
            nn.Dense(cfg.enc_dims//4,2,has_bias=False),
        )
        self.gather=P.Gather()
        self.logsoftmax=P.LogSoftmax(-1)
        self.nll_loss=nn.NLLLoss()
    def construct(self,gene_id,label):
        gene_emb=self.gather(self.gene_emb,gene_id,0)
        func_pred=self.mlp(gene_emb)
        loss=self.nll_loss(self.logsoftmax(func_pred),label)
        if self.training:
            return loss
        else:
            return loss,func_pred,label