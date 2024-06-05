import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from mindspore.common.initializer import initializer
class lora_block(nn.Cell):
    def __init__(self,in_dims,out_dims,hid_dims=16,shard=None):
        super().__init__()
        self.dim=in_dims
        self.alpha=1
        self.A=nn.Dense(in_dims,hid_dims,has_bias=False,weight_init='normal')
        self.B=nn.Dense(hid_dims,out_dims,has_bias=False,weight_init='zeros')
        self.mm=P.MatMul()
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
    def set_shard(self,dp,mp):
        self.A.matmul.shard(((dp,1),(1,1)))
        self.B.matmul.shard(((dp,1),(1,1)))
    def update_weight(self):
        wa=self.A.weight.astype(ms.float32)
        wb=self.B.weight.astype(ms.float32)
        return self.mm(wb,wa)*self.alpha
    def construct(self, x):
        x=self.A(x)
        x=self.B(x)
        return x*self.alpha
        