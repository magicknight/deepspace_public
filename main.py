"""
__author__ = "Zhihua Liang"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""
import gc
import importlib
from commontools.setup import config, logger


def run():
    # Create the Agent then run it..
    mod_name, func_name = config.deepspace.agent.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    agent = getattr(mod, func_name)()
    agent.run()
    agent.finalize()


def main():
    if config.deepspace.device == 'tpu':
        import torch_xla.distributed.xla_multiprocessing as xmp
        gc.collect()
        FLAGS = {}
        xmp.spawn(run, args=(FLAGS,), nprocs=config.deepspace.tpu_cores, start_method='spawn')
    else:
        run()


if __name__ == '__main__':
    main()
