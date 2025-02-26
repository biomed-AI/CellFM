import time
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as ms_np
import mindspore.ops.operations as P
# from GLA import GLALayer
# from minder import AttentionLayer
# from informer import InformerLayer
# from performer import PerformerLayer
from cellot_model.networks.retention import *
from mindspore.common.initializer import initializer, XavierNormal

class ValueEncoder(nn.Cell):
    def __init__(self,emb_dims,shard=None):
        super().__init__()
        self.value_enc=FFN(1,emb_dims)
        self.gather=P.Gather()
        self.one=P.Ones()
        self.add=P.Add()
        self.mul1=P.Mul()
        self.mul2=P.Mul()
        self.mask_emb=ms.Parameter(initializer('zeros',[1,1,emb_dims]))
        self.split=P.Split(-1,2)
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
    def set_shard(self,dp,mp):
        self.w1.matmul.shard(((dp,1),(1,1)))
        self.act1.relu.shard(((dp,1),))
        self.w2.matmul.shard(((dp,1),(1,1)))
        self.act2.relu.shard(((dp,1),))
        self.elu.shard(((dp,1),))
        self.tanh.shard(((dp,1),))
        self.sigmoid.shard(((dp,1),))
        self.add.shard(((dp,1),()))
        self.w3.matmul.shard(((dp,1),(1,1)))
    def construct(self,x):
        b,l=x.shape[:2]
        if len(x.shape)==3:
            unmask,expr=self.split(x)
            unmasked=self.mul1(self.value_enc(expr),unmask)
            masked=self.mul2(self.mask_emb,(1-unmask))
            expr_emb=self.add(masked,unmasked)
        else:
            expr=x.reshape(b,l,1)
            unmask=self.one(expr.shape,expr.dtype)
            expr_emb=self.value_enc(expr)
        return expr_emb,unmask
    
class FFN(nn.Cell):
    def __init__(self,in_dims,emb_dims,b=256,shard=None):
        super().__init__()
        self.w1=nn.Dense(in_dims,b,has_bias=False)
        self.act1=nn.LeakyReLU()
        self.w3=nn.Dense(b,b,has_bias=False)
        self.softmax=P.Softmax(-1)
        self.table=nn.Dense(b,emb_dims,has_bias=False)
        self.dim=emb_dims
        self.add=P.Add()
        self.mul=P.Mul()
        self.a=ms.Parameter(initializer('zeros',[1,1]))
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
    def set_shard(self,dp,mp):
        self.w1.matmul.shard(((dp,1),(1,1)))
        self.act1.select_op.shard(((dp,1),(dp,1)))
        self.add.shard(((dp,1),(dp,1)))
        self.mul.shard(((dp,1),(1,1)))
        self.w3.matmul.shard(((dp,1),(1,1)))
        self.softmax.shard(((dp,1),))
        self.table.matmul.shard(((dp,1),(1,1)))
    def construct(self,x):
        b,l,d=x.shape
        v=P.Reshape()(x,(-1,d))
        v=self.act1(self.w1(v))
        v=self.add(self.w3(v),self.mul(v,self.a))
        v=self.softmax(v)
        v=self.table(v)
        v=P.Reshape()(v,(b,l,-1))
        return v
    
class ValueDecoder(nn.Cell):
    def __init__(self,emb_dims,dropout,zero=False,shard=None):
        super().__init__()
        self.zero=zero
        self.sigmoid=P.Sigmoid()
        self.w1=nn.Dense(emb_dims,emb_dims,has_bias=False)
        self.act=nn.LeakyReLU()
        self.w2=nn.Dense(emb_dims,1,has_bias=False)
        self.relu=P.ReLU()
        if self.zero:
            self.zero_logit = nn.SequentialCell(
                nn.Dense(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Dense(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Dense(emb_dims, 1),
                nn.Sigmoid(),
            )
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
    def set_shard(self,dp,mp):
        self.w1.matmul.shard(((dp,1),(1,1)))
        self.act.select_op.shard(((dp,1),(dp,1)))
        self.w2.matmul.shard(((dp,1),(1,1)))
        if self.zero:
            self.zero_logit[0].matmul.shard(((dp,1),(1,1)))
            self.zero_logit[0].bias_add.shard(((dp,1),(1,)))
            self.zero_logit[1].select_op.shard(((dp,1),(dp,1)))
            self.zero_logit[2].matmul.shard(((dp,1),(1,1)))
            self.zero_logit[2].bias_add.shard(((dp,1),(1,)))
            self.zero_logit[3].select_op.shard(((dp,1),(dp,1)))
            self.zero_logit[4].matmul.shard(((dp,1),(1,1)))
            self.zero_logit[4].bias_add.shard(((dp,1),(1,)))
            self.zero_logit[5].sigmoid.shard(((dp,1),))
    def construct(self,expr_emb):
        b,l,d=expr_emb.shape
        x=self.w2(self.act(self.w1(expr_emb)))
        pred=P.Reshape()(x,(b,l))
        if not self.zero:
            return pred
        else:
            zero_prob=self.zero_logit(expr_emb).reshape(b,-1)
            return pred,zero_prob