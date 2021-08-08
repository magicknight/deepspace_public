
import numpy as np
import importlib
import torch
from commontools.setup import config
shared_array = None


def load_npy(path):
    return np.load(path, allow_pickle=True)


def padding(data):
    return np.pad(data, [(0, 0), (0, config.deepspace.image_size[0] - data.shape[1]), (0, config.deepspace.image_size[1] - data.shape[2])])


def shared_data():
    """load shared data into memory for dataset useage
    """
    if 'shared_dataset' in config.deepspace and config.deepspace.shared_dataset:
        global shared_array
        if isinstance(config.deepspace.train_dataset, list):
            shared_array = list(map(load_npy, config.deepspace.train_dataset))
        else:
            shared_array = np.load(config.deepspace.train_dataset,  allow_pickle=True)
        if 'padding' in config.deepspace and config.deepspace.padding:
            shared_array = list(map(padding, shared_array))
        if 'shared_recon_data_shape' in config.deepspace:
            recon_data = np.zeros(config.deepspace.shared_recon_data_shape, dtype=np.float32)
            shared_array.append(recon_data)


def run():
    # Create the Agent then run it..
    mod_name, func_name = config.deepspace.agent.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    agent = getattr(mod, func_name)(shared_array=shared_array)
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
        # From here on out we are in TPU context
        shared_data()
        FLAGS = {}
        xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=config.deepspace.world_size, join=True, daemon=False, start_method='fork')
    else:
        run()


if __name__ == '__main__':
    main()
