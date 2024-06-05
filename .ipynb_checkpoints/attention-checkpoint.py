import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from mindspore.common.initializer import initializer, XavierUniform

class FullAttention(nn.Cell):
    def __init__(self,emb_dims,num_heads,dropout=0.):
        super().__init__()
        self.q_proj=nn.Dense(emb_dims, emb_dims, has_bias=True, weight_init=XavierUniform())
        self.k_proj=nn.Dense(emb_dims, emb_dims, has_bias=True, weight_init=XavierUniform())
        self.v_proj=nn.Dense(emb_dims, emb_dims, has_bias=False, weight_init=XavierUniform())
        self.to_out=nn.Dense(emb_dims, emb_dims, has_bias=False, weight_init=XavierUniform())
        self.softmax=nn.Softmax(-1)
        self.num_heads=num_heads
        self.head_dim=emb_dims//num_heads
        self.inf = ms.Tensor([-1e5,])
        self.one = ms.Tensor([1.0,])
        self.scale = ms.Tensor([self.head_dim**-0.5],ms.float32)
        self.mul = P.Mul()
        self.posmul = P.Mul()
        self.fill = P.MaskedFill()
        self.fill_m = P.Mul()
        self.fill_s = P.Sub()
        self.fill_a = P.Add()
        self.transpose1 = P.Transpose()
        self.transpose2 = P.Transpose()
        self.QK_matmul = P.BatchMatMul(transpose_b=True)
        self.QKV_matmul = P.BatchMatMul(transpose_b=False)
        self.dropout = nn.Dropout(p=dropout)
    def construct(self, x, y=None, attn_mask=None,k_pos=None,v_pos=None):
        """FullAttention"""
        b,l1,d=x.shape
        h=self.num_heads
        q = self.transpose1(P.Reshape()(self.q_proj(x),(-1,l1,h,self.head_dim)),(0,2,1,3))
        if y is None:
            y=x
        b,l2,_=y.shape
        k = self.transpose1(P.Reshape()(self.k_proj(y),(-1,l2,h,self.head_dim)),(0,2,1,3))
        v = self.transpose1(P.Reshape()(self.v_proj(y),(-1,l2,h,self.head_dim)),(0,2,1,3))
        
        if k_pos is not None:
            k=self.posmul(k,k_pos)
        if v_pos is not None:
            v=self.posmul(v,v_pos)
        scores = self.QK_matmul(q, k)
        scores = self.mul(scores.astype(ms.float32),self.scale)
        if attn_mask is not None:
            attn_mask=attn_mask.reshape(b,1,1,-1)
            attn_mask=self.fill_s(self.one,P.Cast()(attn_mask,scores.dtype))
            # scores=self.fill(scores, ~attn_mask, -math.inf)
            scores=self.fill_a(scores, self.fill_m(attn_mask,self.inf))
        attn = self.softmax(scores).astype(x.dtype)
        attn = self.dropout(attn)
        o = self.transpose2(self.QKV_matmul(attn, v),(0,2,1,3))
        o = self.to_out(P.Reshape()(o,(-1,l1,d)))
        return o
class ffn(nn.Cell):
    def __init__(self,emb_dims,dropout=0.):
        super().__init__()
        self.dense1=nn.Dense(emb_dims, 2*emb_dims)
        self.act=nn.LeakyReLU()
        self.dense2=nn.Dense(2*emb_dims, emb_dims)
        self.dropout = nn.Dropout(p=dropout)
    def construct(self,x):
        b,l,d=x.shape
        x=P.Reshape()(x,(-1,d))
        x=self.dense1(x)
        x=self.act(x)
        x=self.dense2(x)
        x=self.dropout(x)
        return P.Reshape()(x,(b,l,d))
class AttentionLayer(nn.Cell):
    def __init__(self,emb_dims,num_heads,dropout=0.,recompute=False):
        super().__init__()
        self.attn=FullAttention(emb_dims,num_heads,dropout)
        self.ffn=ffn(emb_dims,dropout)
        self.add=P.Add()
        self.norm1=nn.LayerNorm((emb_dims,), epsilon=1e-7)
        self.norm2=nn.LayerNorm((emb_dims,), epsilon=1e-7)
        if recompute:
            self.attn.recompute()
            self.ffn.recompute()
    def construct(self,x,**kwargs):
        attn_mask=kwargs.get('attn_mask')
        y=kwargs.get('y')
        v_pos=kwargs.get('v_pos')
        k_pos=kwargs.get('k_pos')
        x=self.norm1(self.add(x,self.attn(x,y,attn_mask,k_pos,v_pos)))
        x=self.norm2(self.add(x,self.ffn(x)))
        return x