import time
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as ms_np
import mindspore.ops.operations as P
from attention import AttentionLayer
from retention import *
from loss_function import *
from mindspore.dataset.transforms import PadEnd
from mindspore.common.initializer import initializer, XavierNormal
class CellFM(nn.Cell):
    def __init__(self,n_genes,cfg,**kwargs):
        super().__init__()
        # const
        self.depth=cfg.enc_nlayers
        self.if_cls=cfg.label
        self.n_genes=n_genes
        self.add_zero=cfg.add_zero and not cfg.pad_zero
        self.pad_zero=cfg.pad_zero
        # tensor
        self.gene_emb=ms.Parameter(
            initializer(XavierNormal(0.5),[n_genes+1+(-n_genes-1)%8,cfg.enc_dims])
        )
        self.cls_token=ms.Parameter(initializer(XavierNormal(0.5),[1,1,cfg.enc_dims]))
        self.zero_emb=ms.Parameter(initializer('zeros',[1,1,cfg.enc_dims]))
        self.gene_emb[0,:]=0
        # layer
        self.value_enc=ValueEncoder(cfg.enc_dims)
        self.encoder=nn.CellList([
            RetentionLayer(
                cfg.enc_dims,cfg.enc_num_heads,cfg.enc_nlayers,
                cfg.enc_dropout*i/cfg.enc_nlayers, cfg.lora,cfg.recompute
            )
            for i in range(cfg.enc_nlayers)
        ])
        self.value_dec=ValueDecoder(cfg.enc_dims,cfg.dropout,zero=self.add_zero)
        self.cellwise_dec=CellwiseDecoder(cfg.enc_dims,cfg.enc_dims,zero=self.add_zero)
        if cfg.label:
            cls_weight=kwargs.get('cls_weight',np.ones(cfg.num_cls))
            self.weight=ms.Tensor(cls_weight,ms.float32)
            self.cluster_emb=ms.Parameter(
                initializer(XavierNormal(0.5),[cfg.num_cls,cfg.enc_dims])
            )
            self.query=RetentionLayer(
                cfg.enc_dims,cfg.enc_num_heads,0.5,
                0,0,False,shard=shard
            )
            self.classifier=nn.Dense(cfg.enc_dims,1,has_bias=False)
            self.proj=nn.SequentialCell(
                nn.Dense(cfg.enc_dims,cfg.enc_dims),
                nn.LeakyReLU(),
                nn.Dense(cfg.enc_dims,cfg.enc_dims),
                nn.LeakyReLU(),
                nn.Dense(cfg.enc_dims,cfg.enc_dims),
            )
        # operator
        self.mm=P.MatMul(transpose_b=True)
        self.norm=SRMSNorm(cfg.enc_dims)
        self.one=P.Ones()
        self.zero=P.Zeros()
        self.tile=P.Tile()
        self.gather=P.Gather()
        self.gather2=P.Gather()
        self.maskmul=P.Mul()
        self.mul=P.Mul()
        self.add=P.Add()
        self.mean=P.ReduceMean()
        self.posa=P.Add()
        self.rsqrt=P.Rsqrt()
        self.cat1=P.Concat(1)
        self.cat2=P.Concat(1)
        self.slice=P.Slice()
        self.slc=P.Slice()
        self.sum=P.ReduceSum(True)
        self.detach=P.StopGradient()
        self.logsoftmax=P.LogSoftmax(-1)
        # loss
        self.reconstruct1=MaskedMSE(tag='_ge')
        self.reconstruct2=MaskedMSE(tag='_ce')
        self.nll_loss=ops.NLLLoss()
        self.logger=ops.ScalarSummary()
    def encode(self,expr,gene,zero_idx):
        b,l=gene.shape
        gene_emb=self.gather(self.gene_emb,gene,0)
        expr_emb,unmask=self.value_enc(expr)
        len_scale=self.detach(self.rsqrt(self.sum(zero_idx,-1)-1).reshape(b,1,1,1))
        if not self.pad_zero:
            zero_unmask=(1-zero_idx).reshape(b,-1,1)*unmask
            expr_emb=zero_unmask*self.zero_emb+(1-zero_unmask)*expr_emb
        
        expr_emb=self.posa(gene_emb,expr_emb)
        cls_token=self.tile(self.cls_token,(b,1,1))
        expr_emb=self.cat1((cls_token,expr_emb))
        if self.pad_zero:
            expr_emb=self.maskmul(expr_emb,zero_idx.reshape(b,-1,1))
        mask_pos=self.cat2((self.one((b,1,1),unmask.dtype),unmask)).reshape(b,1,-1,1)
        for i in range(self.depth//2):
            expr_emb=self.encoder[i](
                expr_emb,
                v_pos=len_scale,
                attn_mask=mask_pos
            )
        if self.pad_zero:
            mask_pos=zero_idx.reshape(b,1,-1,1)
        else:
            mask_pos=None
        for i in range(self.depth//2,self.depth):
            expr_emb=self.encoder[i](
                expr_emb,
                v_pos=len_scale,
                attn_mask=mask_pos
            )
        return expr_emb,gene_emb
    def forward(self,expr,gene,zero_idx):
        b,l=gene.shape
        emb,gene_emb=self.encode(expr,gene,zero_idx)
        cls_token,expr_emb=emb[:,0],emb[:,1:]
        cls_token=cls_token.reshape(b,-1)
        return expr_emb,gene_emb,cls_token
    def construct(
        self,raw_nzdata,masked_nzdata,nonz_gene,mask_gene,zero_idx,*args
    ):
        expr_emb,gene_emb,cls_token=self.forward(
            masked_nzdata,nonz_gene,zero_idx
        )
        b,l,d=expr_emb.shape
        if self.if_cls:
            attn_mask=self.slice(zero_idx,(0,1),(-1,-1))
            clst_emb=self.cluster_emb.reshape(1,-1,d)
            cluster=self.query(clst_emb,y=expr_emb,attn_mask=attn_mask.reshape(b,1,-1,1))
            labelpred1=self.classifier(cluster).reshape(b,-1)
            labelpred2=self.mm(
                self.proj(cls_token),
                self.cluster_emb.astype(cls_token.dtype)
            )
        if self.add_zero:
            gw_pred,z_prob1=self.value_dec(expr_emb)
            cw_pred,z_prob2=self.cellwise_dec(cls_token,gene_emb)
        else:
            gw_pred=self.value_dec(expr_emb)
            cw_pred=self.cellwise_dec(cls_token,gene_emb)
        if self.training:
            mask=mask_gene
            loss=0
            loss1=self.reconstruct1(gw_pred,raw_nzdata,mask)
            loss2=self.reconstruct2(cw_pred,raw_nzdata,mask)
            loss=loss+loss1+loss2
            if self.add_zero:
                nonz_pos=zero_idx
                loss3=self.bce_loss1(z_prob1,nonz_pos,mask_gene)
                loss4=self.bce_loss2(z_prob2,nonz_pos,mask_gene)
                loss=loss+loss3+loss4
            if self.if_cls:
                label=args[-1]
                logits1=self.logsoftmax(labelpred1.astype(ms.float32))
                logits2=self.logsoftmax(labelpred2.astype(ms.float32))
                loss5=self.nll_loss(logits1,label,self.weight.astype(ms.float32))[0]
                loss6=self.nll_loss(logits2,label,self.weight.astype(ms.float32))[0]
                self.logger('gw_celoss',loss5)
                self.logger('cw_celoss',loss6)
                loss=loss+loss5+loss6
            return loss
        else:
            return gw_pred,cw_pred

class ValueEncoder(nn.Cell):
    def __init__(self,emb_dims):
        super().__init__()
        self.value_enc=FFN(1,emb_dims)
        self.gather=P.Gather()
        self.one=P.Ones()
        self.add=P.Add()
        self.mul1=P.Mul()
        self.mul2=P.Mul()
        self.mask_emb=ms.Parameter(initializer('zeros',[1,1,emb_dims]))
        self.split=P.Split(-1,2)
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
    def __init__(self,in_dims,emb_dims,b=256):
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
    def __init__(self,emb_dims,dropout,zero=False):
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
    def construct(self,expr_emb):
        b,l,d=expr_emb.shape
        x=self.w2(self.act(self.w1(expr_emb)))
        pred=P.Reshape()(x,(b,l))
        if not self.zero:
            return pred
        else:
            zero_prob=self.zero_logit(expr_emb).reshape(b,-1)
            return pred,zero_prob
class CellwiseDecoder(nn.Cell):
    def __init__(self,in_dims,emb_dims=None,dropout=0.,zero=False):
        super().__init__()
        emb_dims=emb_dims or in_dims
        self.act=P.Sigmoid()
        self.sigmoid=P.Sigmoid()
        self.add=P.Add()
        self.tile=P.Tile()
        self.cat=P.Concat(-1)
        self.map=nn.Dense(in_dims, emb_dims,has_bias=False)
        self.bmm=P.BatchMatMul(transpose_b=False)
        self.mm=P.MatMul(transpose_b=True)
        self.relu=P.ReLU()
        self.zero=zero
        if zero:
            self.zero_logit = nn.Dense(emb_dims, emb_dims)
    def construct(self,cell_emb,gene_emb):
        b=cell_emb.shape[0]
        query=self.act(self.map(gene_emb))
        key=cell_emb.reshape(b,-1,1)
        pred=self.bmm(query,key).reshape(b,-1)
        if not self.zero:
            return pred
        else:
            zero_query=self.zero_logit(gene_emb)
            zero_prob=self.sigmoid(self.bmm(zero_query,key)).reshape(b,-1)
            return pred,zero_prob