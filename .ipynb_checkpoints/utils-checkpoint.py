import os
import time
import math
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.scipy as msc
import mindspore.dataset as ds
from tqdm import tqdm,trange
from config import Config
from functools import partial
from scipy.sparse import csr_matrix as csm
from mindspore import nn,ops
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops.operations.math_ops import NPUGetFloatStatusV2, NPUClearFloatStatusV2
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
clip_grad = ops.MultitypeFuncGraph("clip_grad")
grad_scale = ops.MultitypeFuncGraph("grad_scale")
_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()
reciprocal = ops.Reciprocal()

@clip_grad.register("Number", "Tensor", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, -clip_value, clip_value)
    else:
        new_grad = nn.ClipByNorm()(grad, clip_value)
    return new_grad

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class Wrapper(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0, clip_type=0,clip_value=2.0, enable_clip=True):
        super(Wrapper,self).__init__(network, optimizer, sens)
        self.base0 = ms.Tensor(0, ms.int32)
        self.equal = P.Equal()
        self.logic_not = P.LogicalNot()
        self.allreduce = P.AllReduce()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.reduce_all = P.ReduceAll(keep_dims=False)
        self.less_equal = P.LessEqual()
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = ops.Cast()
        self.hyper_map = ops.HyperMap()
        self.clip_type=clip_type
        self.depend = ops.Depend()
        self.clip_value=ms.Tensor([clip_value,])
        self.enable_clip=enable_clip
        self.overflow_logger=ops.ScalarSummary()
    def set_sens(self, value):
        self.sens = value
    @ms.jit
    def clip_grads(self, grads):
        grads = self.hyper_map(ops.partial(clip_grad, self.clip_type, self.clip_value), grads)
        return grads
    def construct(self,*input):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*input)
        sens = self.cast(ops.tuple_to_array((self.sens,)),ms.float32)
        grads = self.grad(self.network, weights)(*input,sens)
        grads = self.grad_reducer(grads)
        grads = self.clip_grads(grads)
        loss = self.depend(loss, self.optimizer(grads))
        return loss
    def start_overflow_check(self, pre_cond, compute_input):
        status = ms.Tensor([0] * 8, ms.int32)
        status = F.depend(status, pre_cond)
        # clear overflow buffer
        clear_status = NPUClearFloatStatusV2()(status)
        compute_input = F.depend(compute_input, clear_status)
        return status, compute_input
    @ms.jit
    def get_overflow_status(self, status, compute_output):
        status = F.depend(status, compute_output)
        get_status = NPUGetFloatStatusV2()(status)

        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(get_status)
            # get_status not equal to [0]*8 means overflow
            flag = self.equal(self.base0, flag_reduce)
            status = F.depend(status, flag)
            # distributed needs to skip allreduce to avoid its overflow affecting the next step
            clear_status = NPUClearFloatStatusV2()(status)
            flag = F.depend(flag, clear_status)
            overall_finite = self.reduce_all(flag)
        else:
            status = F.depend(status, get_status)
            clear_status = NPUClearFloatStatusV2()(status)
            get_status = F.depend(get_status, clear_status)
            flag = self.equal(self.base0, get_status)
            overall_finite = self.reduce_all(flag)
        overflow = self.logic_not(overall_finite)
        return overflow
    
class WrapperWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_update_cell, clip_type=1,clip_value=1.0):
        super(WrapperWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = ops.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, None, self.degree)
        self.loss_scaling_manager = scale_update_cell
        self.loss_scale = ms.Parameter(ms.Tensor(
            scale_update_cell.get_loss_scale(),
            dtype=ms.float32
        ))
        self.clip_type=clip_type
        self.clip_value=ms.Tensor([clip_value,])
        self.scalar_logger=ops.ScalarSummary()
    @ms.jit
    def clip_grads(self, grads):
        grads = self.hyper_map(ops.partial(clip_grad, self.clip_type, self.clip_value), grads)
        return grads
    @ms.jit
    def clip_scale_grads(self, scale, grads):
        grads = self.hyper_map(ops.partial(grad_scale, scale), grads)
        return grads
    def construct(self,*inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.loss_scale
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, ms.float32)
        grads = self.grad(self.network, weights)(*inputs,scaling_sens_filled)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.clip_scale_grads(scaling_sens, grads)
        grads = self.clip_grads(grads)
        
        cond = self.get_overflow_status(status, grads)
        overflow = self.loss_scaling_manager(self.loss_scale, cond)
        self.scalar_logger('overflow',overflow)
        self.scalar_logger('scale',scaling_sens.value())
        if not overflow or scaling_sens.value()==1:
            self.optimizer(grads)
        return loss,overflow,scaling_sens.value()
     
class Adam(nn.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_construct = super().construct
        self.lr_summary = ops.ScalarSummary()
    def construct(self, grads):
        self.lr_summary('lr',self.get_lr())
        return self._original_construct(grads)
    
class AdamWeightDecay(nn.AdamWeightDecay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_construct = super().construct
        self.lr_summary = ops.ScalarSummary()
    def construct(self, grads):
        self.lr_summary('lr',self.get_lr())
        return self._original_construct(grads)
    
class WarmCosineDecay(nn.learning_rate_schedule.LearningRateSchedule):
    def __init__(
        self, 
        current_step,start_lr,
        max_lr, min_lr, 
        warmup_steps, decay_steps
    ):
        super(WarmCosineDecay, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.cur=current_step
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.math_pi = math.pi
        self.delta = 0.5 * (max_lr - min_lr)
        self.cos = P.Cos()
        self.min = P.Minimum()
        self.cast = P.Cast()
    def construct(self, global_step):
        global_step+=self.cur
        if global_step<self.warmup_steps:
            p=self.cast(self.min(global_step, self.warmup_steps), ms.float32)
            lr=self.start_lr+p/self.warmup_steps*(self.max_lr-self.start_lr)
            return lr
        p = self.cast(self.min(global_step-self.warmup_steps, self.decay_steps), ms.float32)
        lr = self.min_lr+self.delta*(1.0+self.cos(self.math_pi*p/self.decay_steps))
        return lr
    
def set_weight_decay(params,weight_decay=1e-5):
    def decay_filter(x):
        name=x.name.lower()
        tag=True
        tag=tag and ('emb' not in name)
        tag=tag and ('layernorm' not in name)
        tag=tag and ('bias' not in name)
        return tag
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [
        {'params': decay_params,'weight_decay': weight_decay}, 
        {'params': other_params,'weight_decay': 0.0}, 
        {'order_params': params}
    ]
    return group_params