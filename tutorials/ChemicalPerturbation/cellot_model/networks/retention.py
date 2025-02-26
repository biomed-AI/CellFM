import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
# from lora import lora_block
from mindspore.common.initializer import initializer,XavierNormal
class SiLU(nn.Cell):
    def __init__(self,strategy=None):
        super().__init__()
        self.sigmoid = P.Sigmoid()
        self.mul = P.Mul()
        if strategy is not None:
            self.set_shard(strategy)
    def set_shard(self, strategy):
        self.sigmoid.shard((strategy,))
        self.mul.shard((strategy, strategy))
    def construct(self, x):
        return self.mul(x,self.sigmoid(x))
    
class SRMSNorm(nn.Cell):
    def __init__(self,emb_dims,strategy=None):
        super().__init__()
        self.scale=ms.Tensor([1/emb_dims**0.5,])
        self.div=P.Div()
        self.norm=P.LpNorm(-1,2,True)
        self.add=P.Add()
        self.eps=ms.Tensor([1e-7,])
        self.mul=P.Mul()
        if strategy is not None:
            self.set_shard(strategy)
    def set_shard(self,strategy):
        self.norm.shard((strategy,))
        self.div.shard((strategy,strategy))
        self.mul.shard((strategy,(1,)))
    def construct(self,x):
        dtype=x.dtype
        x=x.astype(ms.float32)
        x_norm=self.norm(self.mul(x,self.scale))
        return self.div(x,ops.clamp(x_norm,1e-12)).astype(dtype)
class DropPath(nn.Cell):
    def __init__(self, dropout=0.,shard=None):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.mask = ms.Tensor(np.ones((1,1,1)))
        self.tile=P.Tile()
        self.drop_mul=P.Mul()
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
    def set_shard(self,dp,mp):
        self.tile.shard(((1,1,1),))
        self.drop.dropout.shard(((1,1,1),))
        self.drop_mul.shard(((dp,1,1),(dp,1,1)))
    def construct(self, x):
        if not self.training:
            return x
        b,l,d=x.shape
        mask = self.tile(self.mask,(b,1,1))
        mask = self.drop(mask)
        x = self.drop_mul(mask,x)
        return x
    
class Kernel(nn.Cell):
    def __init__(self,shard=None):
        super().__init__()
        self.relu=P.ReLU()
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
    def set_shard(self,dp,mp):
        self.relu.shard(((dp,mp,1,1),))
    def construct(self,x):
        x=self.relu(x)
        return x
    
class MSRetention(nn.Cell):
    def __init__(self,emb_dims,num_heads,lth=None,lora=0,shard=None):
        super().__init__()
        self.num_heads=num_heads
        self.head_dims=emb_dims//num_heads
        self.scale=ms.Tensor([self.head_dims**0.5])
        self.lth=lth
        self.lora=lora
        if self.lora>0:
            self.use_lora(emb_dims,lora,shard)
        self.relu=P.ReLU()
        self.kernelQ=Kernel(shard=shard)
        self.kernelK=Kernel(shard=shard)
        self.kernelV=nn.Identity()
        self.kernelU=nn.SiLU()
        beta=1 if lth is None else (lth*8)**-0.25
        self.q_proj=nn.Dense(emb_dims,emb_dims,has_bias=False,weight_init=XavierNormal(1))
        self.k_proj=nn.Dense(emb_dims,emb_dims,has_bias=False,weight_init=XavierNormal(1))
        self.v_proj=nn.Dense(emb_dims,emb_dims,has_bias=False,weight_init=XavierNormal(beta))
        self.u_proj=nn.Dense(emb_dims,emb_dims,has_bias=False,weight_init=XavierNormal(beta))
        self.o_proj=nn.Dense(emb_dims,emb_dims,has_bias=False,weight_init=XavierNormal(beta))
        self.detach=P.StopGradient()
        self.sum=P.ReduceSum(True)
        self.qkmul=P.BatchMatMul(transpose_b=True)
        self.kvmul=P.BatchMatMul(transpose_a=True)
        self.qkvmul=P.BatchMatMul()
        self.add=P.Add()
        self.mul=P.Mul()
        self.div=P.Div()
        self.split=P.Split(-1,4)
        self.oumul = P.Mul()
        self.ouadd = P.Add()
        self.posmul = P.Mul()
        self.maskmul = P.Mul()
        self.transpose1=P.Transpose()
        self.transpose2=P.Transpose()
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
            self.pre_norm=SRMSNorm(emb_dims,(dp,1,1))
            self.inner_norm=SRMSNorm(self.head_dims,(dp,mp,1,1))
        else:
            self.pre_norm=SRMSNorm(emb_dims)
            self.inner_norm=SRMSNorm(self.head_dims)
    def use_lora(self,emb_dims,r,shard=None):
        self.lora_q=lora_block(emb_dims,emb_dims,r,shard=shard)
        self.lora_k=lora_block(emb_dims,emb_dims,r,shard=shard)
        self.lora_v=lora_block(emb_dims,emb_dims,r,shard=shard)
        self.lora_u=lora_block(emb_dims,emb_dims,r,shard=shard)
        self.lora_o=lora_block(emb_dims,emb_dims,r,shard=shard)
    def combine_weight(self):
        self.q_proj.weight=self.q_proj.weight+self.lora_q.update_weight()
        self.k_proj.weight=self.k_proj.weight+self.lora_k.update_weight()
        self.v_proj.weight=self.v_proj.weight+self.lora_v.update_weight()
        self.u_proj.weight=self.u_proj.weight+self.lora_u.update_weight()
        self.o_proj.weight=self.o_proj.weight+self.lora_o.update_weight()
        self.lora=0
    def lora_compute(self,x,y):
        q_=self.lora_q(x)
        k_=self.lora_k(y)
        v_=self.lora_v(y)
        u_=self.lora_u(x)
        return q_,k_,v_,u_
    def qkvu_compute(self,x,y):
        q=self.q_proj(x)
        k=self.k_proj(y)
        v=self.v_proj(y)
        u=self.u_proj(x)
        if self.lora>0:
            q_,k_,v_,u_=self.lora_compute(x,y)
            q=self.add(q,q_)
            k=self.add(k,k_)
            v=self.add(v,v_)
            u=self.add(u,u_)
        return q,k,v,u
    def set_shard(self,dp,mp):
        mat_shard=((dp,1),(mp,1))
        bias_shard=((dp,mp),(mp,))
        shard_3d=(dp,mp,1)
        shard_4d=(dp,mp,1,1)
        self.pre_norm.layer_norm.shard(((dp,1,1),(1,),(1,)))
        self.add.shard(((dp,1,1),(dp,1,1)))
        self.mul.shard(((dp,mp,1,1),(dp,mp,1,1)))
        self.div.shard(((dp,mp,1,1),(dp,1,1,1)))
        self.transpose1.shard(((dp,1,mp,1),))
        self.transpose2.shard(((dp,mp,1,1),))
        self.split.shard(((dp,1,mp),))
        self.q_proj.matmul.shard(mat_shard)
        self.k_proj.matmul.shard(mat_shard)
        self.v_proj.matmul.shard(mat_shard)
        self.u_proj.matmul.shard(mat_shard)
        self.o_proj.matmul.shard(((dp,mp),(1,mp)))
        self.maskmul.shard((shard_4d,(dp,1,1,1)))
        self.qkmul.shard((shard_4d,shard_4d))
        self.kvmul.shard((shard_4d,shard_4d))
        self.qkvmul.shard((shard_4d,shard_4d))
        self.posmul.shard((shard_4d,(mp,1,1)))
        self.oumul.shard((shard_4d,shard_4d))
        self.ouadd.shard((shard_4d,shard_4d))
    def construct(
        self,x,y=None,v_pos=None,
        attn_mask=None,seq_mask=None
    ):
        h=self.num_heads
        if y is None:
            q,k,v,u=self.qkvu_compute(x,x)
        else:
            q,k,v,u=self.qkvu_compute(x,y)
        _,l1,d=q.shape
        _,l2,d=k.shape
        Q = self.transpose1(P.Reshape()(q,(-1,l1,h,self.head_dims)),(0,2,1,3))
        K = self.transpose1(P.Reshape()(k,(-1,l2,h,self.head_dims)),(0,2,1,3))
        V = self.transpose1(P.Reshape()(v,(-1,l2,h,self.head_dims)),(0,2,1,3))
        U = self.transpose1(P.Reshape()(u,(-1,l1,h,self.head_dims)),(0,2,1,3))
        
        Q=self.kernelQ(Q)
        K=self.kernelK(K)
        U=self.kernelU(U)
        if seq_mask is not None:
            Q=self.maskmul(Q,seq_mask)
        if attn_mask is not None:
            K=self.maskmul(K,attn_mask)
        if v_pos is not None:
            V=self.mul(V,v_pos)
        Q=self.div(Q,self.scale)
        K=self.div(K,self.scale)
        O=self.qkvmul(Q,self.kvmul(K,V))
        O=self.inner_norm(O)
        O=self.oumul(O,U)
        O=P.Reshape()(self.transpose2(O,(0,2,1,3)),(-1,l1,d))
        if self.lora>0:
            O=self.o_proj(O)+self.lora_o(O)
        else:
            O=self.o_proj(O)
        return O
    
class GatedLinearUnit(nn.Cell):
    def __init__(self,emb_dims,lth=None,lora=0,shard=None):
        super().__init__()
        beta=1 if lth is None else (lth*8)**-0.25
        self.u_proj=nn.Dense(emb_dims,emb_dims,has_bias=False,weight_init=XavierNormal(beta))
        self.v_proj=nn.Dense(emb_dims,emb_dims,has_bias=False,weight_init=XavierNormal(beta))
        self.o_proj=nn.Dense(emb_dims,emb_dims,has_bias=False,weight_init=XavierNormal(beta))
        self.uvmul = P.Mul()
        self.lora=lora
        if self.lora>0:
            self.use_lora(emb_dims,lora,shard=shard)
        if shard is not None:
            dp,mp=shard
            strategy=(dp,1,1)
            self.set_shard(dp,mp)
            self.norm=SRMSNorm(emb_dims,(dp,1,1))
        else:
            self.norm=SRMSNorm(emb_dims,)
    def set_shard(self,dp,mp):
        mat_shard=((dp,1),(mp,1))
        bias_shard=((dp,mp),(mp,))
        self.norm.layer_norm.shard(((dp,1,1),(1,),(1,)))
        self.u_proj.matmul.shard(mat_shard)
        self.v_proj.matmul.shard(mat_shard)
        self.uvmul.shard(((dp,mp),(dp,mp)))
        self.o_proj.matmul.shard(((dp,mp),(1,mp)))
    def use_lora(self,emb_dims,r,shard=None):
        self.lora_u=lora_block(emb_dims,emb_dims,r,shard=shard)
        self.lora_v=lora_block(emb_dims,emb_dims,r,shard=shard)
        self.lora_o=lora_block(emb_dims,emb_dims,r,shard=shard)
    def combine_weight(self):
        self.u_proj.weight=self.u_proj.weight+self.lora_u.update_weight()
        self.v_proj.weight=self.v_proj.weight+self.lora_v.update_weight()
        self.o_proj.weight=self.o_proj.weight+self.lora_o.update_weight()
        self.lora=0
    def construct(self,x):
        b,l,d=x.shape
        x=P.Reshape()(x,(-1,d))
        if self.lora>0:
            u=self.u_proj(x)+self.lora_u(x)
            v=self.v_proj(x)+self.lora_v(x)
            o=self.uvmul(u,v)
            o=self.o_proj(o)+self.lora_o(o)
        else:
            u=self.u_proj(x)
            v=self.v_proj(x)
            o=self.uvmul(u,v)
            o=self.o_proj(o)
        x=P.Reshape()(o,(b,l,-1))
        return x
class CrossRetentionLayer(nn.Cell):
    def __init__(self,emb_dims,num_heads,dropout=0.,recompute=False,shard=None):
        super().__init__()
        self.attn1=MSRetention(emb_dims,num_heads,shard=shard)
        self.attn2=MSRetention(emb_dims,num_heads,shard=shard)
        self.ffn=GatedLinearUnit(emb_dims)
        self.add=P.Add()
        self.dropout=nn.Dropout(p=dropout)
        self.post_norm1=nn.LayerNorm((emb_dims,))
        self.post_norm2=nn.LayerNorm((emb_dims,))
        self.post_norm3=nn.LayerNorm((emb_dims,))
        if recompute:
            self.attn1.recompute()
            self.attn2.recompute()
            self.ffn.recompute()
    def construct(self,x,y,**kwargs):
        v_pos=kwargs.get('v_pos')
        seq_mask=kwargs.get('seq_mask')
        attn_mask=kwargs.get('attn_mask')
        out=self.dropout(self.attn1(x))
        x=self.post_norm1(self.add(x,out))
        out=self.dropout(self.attn2(x,y,v_pos,attn_mask,seq_mask))
        x=self.post_norm2(self.add(x,out))
        out=self.dropout(self.ffn(x))
        x=self.post_norm3(self.add(x,out))
        return x
class RetentionLayer(nn.Cell):
    def __init__(self,emb_dims,num_heads,lth,dropout=0.,lora=False,recompute=False,shard=None):
        super().__init__()
        self.attn=MSRetention(emb_dims,num_heads,lth,lora,shard=shard)
        self.ffn=GatedLinearUnit(emb_dims,lth,lora)
        self.dropout=nn.Dropout(p=dropout)
        self.add=P.Add()
        self.mul=P.Mul()
        self.post_norm1=nn.LayerNorm((emb_dims,))
        self.post_norm2=nn.LayerNorm((emb_dims,))
        self.alpha=(2*lth)**0.25
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
        if recompute:
            self.attn.recompute()
            self.ffn.recompute()
    def set_shard(self,dp,mp):
        self.add.shard(((dp,1,1),(dp,1,1)))
        self.mul.shard(((dp,1,1),()))
        self.dropout.dropout.shard(((dp,1,1),))
        self.post_norm1.layer_norm.shard(((dp,1,1),(1,),(1,)))
        self.post_norm2.layer_norm.shard(((dp,1,1),(1,),(1,)))
    def construct(self,x,**kwargs):
        y=kwargs.get('y')
        v_pos=kwargs.get('v_pos')
        seq_mask=kwargs.get('seq_mask')
        attn_mask=kwargs.get('attn_mask')
        out=self.dropout(self.attn(x,y,v_pos,attn_mask,seq_mask))
        x=self.post_norm1(self.add(self.mul(x,self.alpha),out))
        out=self.dropout(self.ffn(x))
        x=self.post_norm2(self.add(self.mul(x,self.alpha),out))
        return x