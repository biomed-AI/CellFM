import sys
sys.path.append('..')
import mindspore as ms
from model import *
class Backbone(nn.Cell):
    def __init__(self,n_genes,cfg,**kwargs):
        super().__init__()
        self.depth=cfg.enc_nlayers
        self.if_cls=cfg.label
        self.n_genes=n_genes
        # tensor
        self.gene_emb=ms.Parameter(
            initializer(XavierNormal(0.5),[n_genes+1+(-n_genes-1)%8,cfg.enc_dims])
        )
        self.cls_token=ms.Parameter(initializer(XavierNormal(0.5),[1,1,cfg.enc_dims]))
        self.gene_emb[0,:]=0
        # layer
        self.value_enc=ValueEncoder(cfg.enc_dims)
        self.encoder=nn.CellList([
            RetentionLayer(
                cfg.enc_dims,cfg.enc_num_heads,cfg.enc_nlayers,
                cfg.enc_dropout*i/cfg.enc_nlayers, cfg.lora,
                cfg.recompute,
            )
            for i in range(cfg.enc_nlayers)
        ])
        self.one=P.Ones()
        self.zero=P.Zeros()
        self.tile=P.Tile()
        self.gather=P.Gather()
        self.maskmul=P.Mul()
        self.posa=P.Add()
        self.rsqrt=P.Rsqrt()
        self.cat1=P.Concat(1)
        self.sum=P.ReduceSum(True)
        self.detach=P.StopGradient()
    def construct(self,expr,gene,zero_idx):
        b,l=gene.shape
        gene_emb=self.gather(self.gene_emb,gene,0)
        expr_emb,unmask=self.value_enc(expr)
        len_scale=self.detach(self.rsqrt(self.sum(zero_idx,-1)-1).reshape(b,1,1,1))

        expr_emb=self.posa(gene_emb,expr_emb)
        cls_token=self.tile(self.cls_token,(b,1,1))
        expr_emb=self.cat1((cls_token,expr_emb))
        expr_emb=self.maskmul(expr_emb,zero_idx.reshape(b,-1,1))
        mask_pos=zero_idx.reshape(b,1,-1,1)
        for i in range(self.depth):
            expr_emb=self.encoder[i](
                expr_emb,
                v_pos=len_scale,
                attn_mask=mask_pos
            )
        return expr_emb
class Net(nn.Cell):
    def __init__(self,backbone,cfg,**kwargs):
        super().__init__()
        # const
        self.num_class=cfg.num_cls
        self.extractor=backbone
        cls_weight=kwargs.get('cls_weight',np.ones(cfg.num_cls))
        self.weight=ms.Tensor(cls_weight,ms.float32)
        self.cluster_emb=ms.Parameter(
            initializer(XavierNormal(0.5),[cfg.num_cls,cfg.enc_dims])
        )
        self.query_layer=nn.CellList([
            CrossRetentionLayer(cfg.enc_dims,cfg.enc_num_heads,cfg.enc_dropout,False)
            for i in range(2)
        ])
        self.classifier=nn.Dense(cfg.enc_dims,1,has_bias=False)
        # operator
        self.tile=P.Tile()
        self.slice=P.Slice()
        self.cat1=P.Concat(1)
        self.mm=P.MatMul(transpose_b=True)
        self.logsoftmax=P.LogSoftmax(-1)
        # loss
        self.nll_loss=ops.NLLLoss()
        self.logger=ops.ScalarSummary()
    def forward(self,expr,gene,zero_idx):
        emb=self.extractor(expr,gene,zero_idx)
        cls_token,expr_emb=emb[:,:1],emb[:,1:]
        b,l,d=expr_emb.shape
        attn_mask=self.slice(zero_idx,(0,1),(-1,-1))
        clst_emb=self.cat1(
            (cls_token.reshape(-1,1,d),
             self.tile(self.cluster_emb.astype(cls_token.dtype),(b,1,1)))
        )
        for query in self.query_layer:
            clst_emb=query(clst_emb,y=expr_emb,attn_mask=attn_mask.reshape(b,1,-1,1))
        cls_token,cluster=clst_emb[:,0],clst_emb[:,1:]
        labelpred1=self.classifier(cluster).reshape(b,-1)
        labelpred2=self.mm(
            cls_token,self.cluster_emb.astype(cls_token.dtype)
        )
        return labelpred1,labelpred2
    def construct(
        self,nonz_data,nonz_gene,zero_idx,label
    ):
        labelpred1,labelpred2=self.forward(
            nonz_data,nonz_gene,zero_idx
        )
        logits1=self.logsoftmax(labelpred1.astype(ms.float32))
        logits2=self.logsoftmax(labelpred2.astype(ms.float32))
        loss1=self.nll_loss(logits1,label,self.weight.astype(ms.float32))[0]
        loss2=self.nll_loss(logits2,label,self.weight.astype(ms.float32))[0]
        self.logger('gw_celoss',loss1)
        self.logger('cw_celoss',loss2)
        loss=loss1+loss2
        if self.training:
            return loss
        else:
            return loss,labelpred1,label