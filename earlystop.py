import os
import copy
import numpy as np
import mindspore as ms
from mindspore import nn,ops
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as Validator
from mindspore.train.serialization import load_param_into_net
from mindspore import log as logger
from mindspore.ops import ReduceOp
from mindspore.communication import get_group_size,get_rank
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.callback._callback import Callback, _handle_loss


_smaller_better_metrics = ['hausdorff_distance', 'mae', 'mse', 'loss', 'perplexity',
                           'mean_surface_distance', 'root_mean_square_distance', 'eval_loss']


class EarlyStopping(Callback):
    def __init__(
        self, monitor='eval_loss', min_delta=0, patience=0,
        verbose=False, mode='auto', baseline=None, min_epoch=0,
        restore_best_weights=False,restore_path=None
    ):
        super(EarlyStopping, self).__init__()
        self.restore_path=restore_path
        if self.restore_path is not None:
            os.makedirs(self.restore_path,exist_ok=True)
        self.monitor = Validator.check_value_type('monitor', monitor, str)
        min_delta = Validator.check_value_type("min_delta", min_delta, [float, int])
        self.min_delta = abs(min_delta)
        self.best_epoch=0
        self.patience = Validator.check_non_negative_int(patience)
        self.verbose = Validator.check_bool(verbose)
        self.mode = Validator.check_value_type('mode', mode, str)
        self.baseline = Validator.check_value_type("baseline", baseline, [float]) if baseline else None
        self.restore_best_weights = Validator.check_bool(restore_best_weights)
        self.min_epoch=Validator.check_value_type("min_epoch", min_epoch, [int])
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights_param_dict = None
        self._reduce = ValueReduce()

        self.parallel_mode = auto_parallel_context().get_parallel_mode()
        self.rank_size = 1 if self.parallel_mode == ParallelMode.STAND_ALONE else get_group_size()
        self.rank_id = 0 if self.parallel_mode == ParallelMode.STAND_ALONE else get_rank()
        if self.mode not in ['auto', 'min', 'max']:
            raise ValueError("mode should be 'auto', 'min' or 'max', but got %s." % self.mode)
        if self.mode == 'min' or (self.mode == 'auto' and self.monitor in _smaller_better_metrics):
            self.is_improvement = lambda a, b: np.less(a, b-self.min_delta)
            self.best = np.Inf
        else:
            self.is_improvement = lambda a, b: np.greater(a, b+self.min_delta)
            self.best = -np.Inf

    def on_train_begin(self, run_context):

        self.wait = 0
        self.stopped_epoch = 0
        if self.mode == 'min' or (self.mode == 'auto' and self.monitor in _smaller_better_metrics):
            self.best = np.Inf
        else:
            self.best = -np.Inf
        self.best_weights_param_dict = None

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()

        cur_epoch = cb_params.get("cur_epoch_num")
        current_value = self._get_monitor_value(cb_params)
        rank_size=self.rank_size
        if rank_size == 1:
            current = current_value
        else:
            current = self._reduce(Tensor(current_value.astype(np.float32))) / rank_size

        if current is None:
            return
        self.wait += 1
        if self.is_improvement(current, self.best):
            self.best = current
            if self.restore_best_weights:
                if self.restore_path is not None:
                    if self.rank_id==0: 
                        # os.remove(self.restore_path+f'/best.ckpt')
                        ms.save_checkpoint(
                            cb_params.train_network.parameters_dict(),
                            self.restore_path+f'/best.ckpt'
                        )
                    self.best_epoch=cur_epoch
                else:
                    self.best_weights_param_dict = copy.deepcopy(cb_params.train_network.parameters_dict())
            if self.baseline is None or self.is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience and cur_epoch>=self.min_epoch:
            self.stopped_epoch = cur_epoch
            run_context.request_stop()
            if self.restore_best_weights:
                if self.verbose:
                    print(f'Restoring model weights from the end of the best epoch {self.best_epoch}.')
                if self.restore_path is not None:
                    self.best_weights_param_dict=ms.load_checkpoint(
                        self.restore_path+f'/best.ckpt'
                    )
                load_param_into_net(cb_params.train_network, self.best_weights_param_dict)

    def on_train_end(self, run_context):
        if self.stopped_epoch > 0 and self.verbose:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))

    def _get_monitor_value(self, cb_params):
        monitor_candidates = {}
        if self.monitor == "loss":
            loss = cb_params.get("net_outputs")
            monitor_value = _handle_loss(loss)
            if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
                logger.warning("Invalid %s.", self.monitor)
        else:
            monitor_candidates = cb_params.get("eval_results", {})
            monitor_value = monitor_candidates.get(self.monitor)

        if monitor_value is None:
            support_keys = set(["loss"] + list(monitor_candidates.keys()))
            logger.warning('Early stopping is conditioned on %s, '
                           'which is not available. Available choices are: %s',
                           self.monitor, support_keys)
        if isinstance(monitor_value, np.ndarray) and monitor_value.shape != ():
            raise ValueError("EarlyStopping only supports scalar monitor now.")
        return np.array(monitor_value) if monitor_value else None


class ValueReduce(nn.Cell):
    """
    Reduces the tensor data across all devices, all devices will get the same final result.
    For more details, please refer to :class:`mindspore.ops.AllReduce`.
    """
    def __init__(self):
        super(ValueReduce, self).__init__()
        self.allreduce = ops.AllReduce(ReduceOp.SUM)

    def construct(self, x):
        return self.allreduce(x).asnumpy()


class ValueReduce(nn.Cell):
    def __init__(self):
        super(ValueReduce, self).__init__()
        self.allreduce = ops.AllReduce(ReduceOp.SUM)

    def construct(self, x):
        return self.allreduce(x).asnumpy()
    
def pearson(pred,target,mask):
    p=[]
    for i in range(len(pred)):
        p.append(sp.stats.pearsonr(pred[i,mask[i]],target[i,mask[i]])[0])
    p=np.array(p)
    return np.nanmean(p)