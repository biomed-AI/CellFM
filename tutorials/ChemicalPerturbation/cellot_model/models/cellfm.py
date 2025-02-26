import os
import sys
import pandas as pd
import mindspore as ms
from mindspore import nn,ops
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer, XavierNormal
from cellot_model.networks.scret import *

class CellwiseDecoder2(nn.Cell):
    def __init__(self,in_dims,emb_dims=None,dropout=0.,zero=False,shard=None):
        super().__init__()
        emb_dims=emb_dims or in_dims
        self.act=P.Sigmoid()
        self.sigmoid=P.Sigmoid()
        self.add=P.Add()
        self.tile=P.Tile()
        self.cat=P.Concat(-1)
        # self.map=nn.Dense(in_dims, emb_dims,has_bias=False)

        self.map=nn.SequentialCell(
            nn.Dense(emb_dims,512,has_bias=False),
            nn.Dropout(p=0.15),
            nn.SiLU(),
            nn.Dense(512,798,has_bias=False),
        )

        self.bmm=P.BatchMatMul(transpose_b=False)
        self.mm=P.MatMul(transpose_b=True)
        self.relu=P.ReLU()
        self.zero=zero
        if zero:
            self.zero_logit = nn.Dense(emb_dims, emb_dims)
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
    def set_shard(self,dp,mp):
        self.map.matmul.shard(((dp,1),(1,1)))
        self.act.shard(((dp,1,1),))
        self.sigmoid.shard(((dp,1,1),))
        self.bmm.shard(((dp,1,1),(dp,1,1)))
        if self.zero:
            self.zero_logit.matmul.shard(((dp,1),(1,1)))
            self.zero_logit.bias_add.shard(((dp,1),(1,)))
    def construct(self,cell_emb):
        # b=cell_emb.shape[0]
        # query=self.act(self.map(gene_emb))
        # key=cell_emb.reshape(b,-1,1)
        # pred=self.bmm(query,key).reshape(b,-1)
        # if not self.zero:
        #     return pred
        # else:
        #     zero_query=self.zero_logit(gene_emb)
        #     zero_prob=self.sigmoid(self.bmm(zero_query,key)).reshape(b,-1)
        #     return pred,zero_prob

        pred = self.map(cell_emb)
        return pred

class backbone(nn.Cell):
    def __init__(self,n_genes,cfg,shard=None,**kwargs):
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
        self.ST_emb=ms.Parameter(
            initializer(XavierNormal(0.5),[1,2,cfg.enc_dims])
        )
        self.cls_token=ms.Parameter(initializer(XavierNormal(0.5),[1,1,cfg.enc_dims]))
        self.zero_emb=ms.Parameter(initializer('zeros',[1,1,cfg.enc_dims]))
        # self.platform_emb=ms.Parameter(initializer('zeros',[cfg.platforms+1,cfg.enc_dims]))
        self.gene_emb[0,:]=0
        # layer
        self.value_enc=ValueEncoder(cfg.enc_dims,shard=shard)
        self.ST_enc=FFN(1,cfg.enc_dims,shard=shard)
        self.encoder=nn.CellList([
            RetentionLayer(
                cfg.enc_dims,cfg.enc_num_heads,cfg.enc_nlayers,
                cfg.enc_dropout*i/cfg.enc_nlayers, cfg.lora,
                cfg.recompute,shard=shard
            )
            for i in range(cfg.enc_nlayers)
        ])
        self.value_dec=ValueDecoder(cfg.enc_dims,cfg.dropout,zero=self.add_zero,shard=shard)
        self.cellwise_dec=CellwiseDecoder2(50,zero=self.add_zero)
        self.proj2=nn.SequentialCell(
            nn.Dense(1536,512),
            nn.SiLU(),
            nn.Dense(512,50),
            # nn.LeakyReLU(),
            # nn.Dense(cfg.enc_dims,cfg.enc_dims),
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
        # self.split=P.SplitV([1,2,-1],1,3)
        self.logsoftmax=P.LogSoftmax(-1)
        # loss
        # self.sim=Similarity(cfg.sim)
        # shard
        self.logger=ops.ScalarSummary()
        # st=None
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
            st=(dp,1,1)
    def set_shard(self,dp,mp):
        self.mm.shard(((dp,1),(1,1)))
        self.tile.shard(((1,1,1),))
        self.rsqrt.shard(((dp,1),))
        self.gather.shard(((mp,1),(dp,1)))
        self.expand.shard(((dp,1),))
        self.maskmul.shard(((dp,1,1),(dp,1,1)))
        self.sub.shard(((dp,1),(dp,1)))
        self.add.shard(((1,1,1),(dp,1,1)))
        self.posa.shard(((dp,1,1),(dp,1,1)))
        self.cat2.shard(((dp,1,1),(dp,1,1)))
        self.cat1.shard(((dp,1,1),(dp,1,1),(dp,1,1)))
        self.slice.shard(((dp,1,1),))
        self.slc.shard(((dp,1),))
        # self.split.shard(((dp,1,1),))
    def encode(self,expr,gene,zero_idx):
        b,l=gene.shape
        gene_emb=self.gather(self.gene_emb,gene,0)
        expr_emb,unmask=self.value_enc(expr)
        len_scale=self.detach(self.rsqrt(self.sum(zero_idx,-1)-3).reshape(b,1,1,1))
        
        expr_emb=self.posa(gene_emb,expr_emb)
        cls_token=self.tile(self.cls_token,(b,1,1))
        expr_emb=self.cat1((cls_token,expr_emb))
        mask_pos=self.cat2((self.one((b,1,1),unmask.dtype),unmask)).reshape(b,1,-1,1)
        for i in range(self.depth//2):
            expr_emb=self.encoder[i](
                expr_emb,
                v_pos=len_scale,
                attn_mask=mask_pos
            )
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
        cls_token=self.proj2(cls_token)
        return expr_emb,gene_emb,cls_token
    def construct(
        self,raw_nzdata,
        nonz_gene,mask_gene,zero_idx,*args
    ):
        expr_emb,gene_emb,cls_token=self.forward(
            raw_nzdata,nonz_gene,zero_idx
        )
        gw_pred=self.value_dec(expr_emb)
        cw_pred=self.cellwise_dec(cls_token)
        return gw_pred,cw_pred
    
class Config:
    start_lr=1e-7
    max_lr=1e-6
    min_lr=5e-7
    factor=5
    lora=0
    alpha=0
    lamb=10
    nb_features=256
    nonz_len=2048
    mask_len=2048
    filt_len=200
    dropout=0.1
    enc_dims=1536
    enc_nlayers=2
    enc_num_heads=48
    enc_dropout=0.1
    dec_dims=512
    dec_nlayers=6
    dec_num_heads=16
    dec_dropout=0.1
    temp=0.2
    eps=1e-2
    recompute=True
    sim=0.8
    add_zero=False
    pad_zero=True
    label=False
    num_cls=80
    platforms=27
    ttl_step=1e5
    

def load_cellfm_model(drug):
    cfg=Config()
    gene_info_path = '../../csv/expand_gene_info.csv'
    gene_info=pd.read_csv(gene_info_path,index_col=0,header=0)
    geneset={j:i+1 for i,j in enumerate(gene_info.index)}
    model = backbone(len(geneset),cfg)
    check_point_path = f'./checkpoint/sciplex_{drug}.ckpt'
    para=ms.load_checkpoint(check_point_path)
    ms.load_param_into_net(model, para)
    
    return model, geneset