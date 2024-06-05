import scipy as sp
import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
from mindspore import nn,ops
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
class MaskedMSE(nn.Cell):
    def __init__(self,tag=None,shard=None):
        super().__init__()
        self.tag=tag or ''
        self.sub=P.Sub()
        self.sq=P.Square()
        self.cast=P.Cast()
        self.mean=P.ReduceMean(False)
        self.sum=P.ReduceSum(False)
        self.fill_m=P.Mul()
        self.div=P.Div()
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
        # logger
        self.loss_logger=ops.ScalarSummary()
    def set_shard(self,dp,mp):
        self.sub.shard(((dp,1),(dp,1)))
        self.sq.shard(((dp,1),))
        self.mean.shard(((dp,1),))
        self.sum.shard(((dp,1),))
        self.fill_m.shard(((dp,1),(dp,1)))
        self.div.shard(((),()))
    def construct(self,pred,target,mask=None):
        pred = self.cast(pred, ms.float32)
        target = self.cast(target, ms.float32)
        loss=self.sq(self.sub(pred,target))
        if mask is not None:
            mask=self.cast(mask,ms.float32)
            loss=self.sum(self.fill_m(loss,mask))
            num=self.sum(mask)
            loss=self.div(loss,num)
            self.loss_logger(f'MaskedMSE{self.tag}',loss)
            return loss
        loss=self.mean(loss)
        self.loss_logger(f'MSE{self.tag}',loss)
        return loss

class BCE(nn.Cell):
    def __init__(self,tag='',shard=None):
        super().__init__()
        self.tag=tag
        self.sigmoid=P.Sigmoid()
        self.log=P.Log()
        self.gather=P.Gather(1)
        self.cat=P.Concat(-1)
        self.sub=P.Sub()
        self.div=P.Div()
        self.mul=P.Mul()
        self.eps=ms.Tensor([1e-12])
        self.sum1=P.ReduceSum(False)
        self.sum=P.ReduceSum(False)
        self.mean=P.ReduceMean(False)
        if shard is not None:
            dp,mp=shard
            self.set_shard(dp,mp)
        self.loss_logger=ops.ScalarSummary()
    def set_shard(self,dp,mp):
        self.sigmoid.shard(((dp,1),))
        self.sub.shard(((),(dp,1)))
        self.cat.shard(((dp,1),(dp,1)))
        self.log.shard(((dp,),))
        self.mul.shard(((dp,1),(dp,1)))
        self.mean.shard(((dp,),))
    def construct(self,pred,target,mask=None):
        pred=P.Reshape()(pred.astype(ms.float32),(-1,1))
        target=P.Reshape()(target.astype(ms.float32),(-1,1))
        pred=self.cat((self.sub(1,pred),pred))
        pred=self.log(ops.clamp(pred,1e-12,1))
        logit=self.cat((self.sub(1,target),target))
        logit=-self.sum1(self.mul(pred,logit),1)
        if mask is None:
            loss=self.mean(logit)
        else:
            mask=P.Reshape()(mask.astype(ms.float32),(-1,1))
            loss=self.div(self.sum(logit),self.sum(mask))
        self.loss_logger(f'BCE{self.tag}',loss)
        return loss

class NLL_loss(nn.Cell):
    def __init__(self,weight=None,reduction='mean',ignore_index=-100):
        super().__init__()
        self.weight=weight or ms.Tensor([1],ms.float32)
        self.reduction=reduction
        self.ignore_index=ignore_index
        self.gather=P.Gather(1)
        self.sum1=P.ReduceSum()
        self.sum2=P.ReduceSum()
        self.mean=P.ReduceMean()
        self.eq=P.Equal()
        self.loss_logger=ops.ScalarSummary()
    def construct(self,pred,target):
        b,n=pred.shape
        pred=pred.astype(ms.float32)
        nll=self.gather(pred,target,1)
        mask=self.eq(target,self.ignore_index)
        mask=(1-mask).astype(ms.float32)
        nll=-mask*nll*self.weight
        if self.reduction=='sum':
            loss=self.sum1(nll)
            self.loss_logger('nll_loss',loss)
        elif self.reduction=='mean':
            loss=self.sum1(nll)/ops.clamp(self.sum2(mask),1)
            self.loss_logger('nll_loss',loss)
        else:
            loss=nll
        return loss