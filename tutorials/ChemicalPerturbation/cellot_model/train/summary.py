from pathlib import Path

import pandas as pd
from datetime import datetime


class Logger:
    def __init__(self, path, max_items=1024):
        train_path = Path(path / 'train_log.txt')
        eval_path = Path(path / 'eval_log.txt')
        
        self.path = {'train':train_path, 'eval':eval_path}
        
        current_time = datetime.now()
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

        for key in ['train', 'eval']:
            with open(self.path[key], 'a') as file:
                file.write('-'*10 + '\n')
                file.write(time_string+ '\n')
                
        
    def log(self, key, **kwargs):
        # step = 'step: {} '.format(kwargs['step'])
        
        # kwargs = {k: float(v) for k, v in kwargs.items() if k != 'step'}
        output_text = ['{}: {}'.format(k, v) for k,v in kwargs.items()]
        output_text = ' '.join(output_text)
        
        with open(self.path[key], 'a') as file:
            file.write(output_text + '\n')
            
        
    def flush(self):
        pass

# class Logger:
#     def __init__(self, path, max_items=1024):
#         self.path = Path(path)
#         self.max_items = max_items
#         self.loggers = dict()

#         self.path.parent.mkdir(exist_ok=True, parents=True)

#     def log(self, key, item=None, **kwargs):
#         if key not in self.loggers:
#             self.loggers[key] = SingleLogger(
#                 self.path, key, max_items=self.max_items
#             )

#         logger = self.loggers[key]
#         logger.log(item, **kwargs)

#     def flush(self):
#         for logger in self.loggers.values():
#             logger.flush()
#         return


# class SingleLogger:
#     def __init__(self, path, key='scalars', max_items=1024):
#         self.path = path
#         self.max_items = max_items
#         self.key = key

#         self.store = list()
#         self.curr_step = None
#         self.curr_item = dict()

#     def _clear_curr_item(self):
#         if len(self.curr_item) == 0:
#             return

#         self.store.append(self.curr_item.copy())
#         self.curr_item = dict()

#         if len(self.store) > self.max_items:
#             self.flush()

#         return

#     def log(self, item=None, **kwargs):
#         item = {} if item is None else item
#         kwargs.update(item)
#         step = kwargs['step']
#         kwargs = {k: float(v) for k, v in kwargs.items()}

#         if step != self.curr_step:
#             self._clear_curr_item()
#             self.curr_step = step

#         self.curr_item.update(kwargs)

#         return

#     def _write(self):
#         df = pd.DataFrame(self.store).set_index('step')
#         # df.to_hdf(self.path, self.key, append=True, format='table')
#         df.to_csv(self.path, mode='a')
#         return

#     def flush(self, key=None):
#         self._clear_curr_item()
#         self._write()
#         self.store = list()
