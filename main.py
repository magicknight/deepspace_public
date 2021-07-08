'''
 ┌─────────────────────────────────────────────────────────────┐
 │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
 ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
 │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
 ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
 │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
 ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
 │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
 ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
 │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
 │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
 │      └───┴─────┴───────────────────────┴─────┴───┘          │
 └─────────────────────────────────────────────────────────────┘

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-07-08 14:50:44
LastEditors: Zhihua Liang
LastEditTime: 2021-07-08 14:50:45
FilePath: /home/zhihua/framework/deepspace/main.py
'''

'''
 ┌─────────────────────────────────────────────────────────────┐
 │┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
 ││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
 │├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
 ││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
 │├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
 ││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
 │├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
 ││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
 │└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
 │      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
 │      └───┴─────┴───────────────────────┴─────┴───┘          │
 └─────────────────────────────────────────────────────────────┘
__author__ = "Zhihua Liang"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent

Description: 
Author: Zhihua Liang
Github: https://github.com/magicknight
Date: 2021-07-08 09:15:09
LastEditors: Zhihua Liang
LastEditTime: 2021-07-08 09:15:09
FilePath: /home/zhihua/framework/deepspace/main.py
'''


import numpy as np
import importlib
import torch
from commontools.setup import config
shared_array = None


def run():
    # Create the Agent then run it..
    mod_name, func_name = config.deepspace.agent.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    if shared_array is not None:
        agent = getattr(mod, func_name)(shared_array=shared_array)
    else:
        agent = getattr(mod, func_name)()
    agent.run()
    agent.finalize()


def _mp_fn(index, args):
    torch.set_default_tensor_type('torch.FloatTensor')
    # suppress_output(xm.is_master_ordinal())
    run()


def main():
    # if running on tpu
    if config.deepspace.device == 'tpu':
        # import the distributed tools from torch_xla.
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch.multiprocessing as mp
        if config.deepspace.shared_dataset:
            global shared_array
            shared_array = np.load(config.deepspace.train_dataset,  allow_pickle=True)
            # shared_array = np.ones((256, 256, 256))
        # From here on out we are in TPU context
        FLAGS = {}
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=config.deepspace.world_size, join=True, daemon=False, start_method='fork')
    else:
        run()


if __name__ == '__main__':
    main()
