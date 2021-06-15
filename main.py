"""
__author__ = "Zhihua Liang"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""
import importlib
import torch
from commontools.setup import config
# from deepspace.utils.distribute import suppress_output


def run():
    # Create the Agent then run it..
    mod_name, func_name = config.deepspace.agent.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    agent = getattr(mod, func_name)()
    agent.run()
    agent.finalize()


def _mp_fn(index, args):
    import dis
    import torch_xla.core.xla_model as xm
    torch.set_default_tensor_type('torch.FloatTensor')
    # suppress_output(xm.is_master_ordinal())
    run()


def main():
    # if running on tpu
    if config.deepspace.device == 'tpu':
        # import the distributed tools from torch_xla.
        import torch_xla
        import torch_xla.debug.metrics as met
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.utils.utils as xu
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        # From here on out we are in TPU context
        FLAGS = {}
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=config.deepspace.world_size, join=True, daemon=False, start_method='fork')
    else:
        run()


if __name__ == '__main__':
    main()
