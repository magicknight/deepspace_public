from deepspace.graphs.scheduler.warmup import WarmupAndExponentialDecayScheduler


def wrap_optimizer_with_scheduler(optimizer,
                                  scheduler_type=None,
                                  scheduler_divisor=None,
                                  scheduler_divide_every_n_epochs=None,
                                  num_steps_per_epoch=None,
                                  summary_writer=None):
    """Wraps an optimizer in a `torch.optim.lr_scheduler` object.
    Args:
      optimizer: Instance of `torch.optim.Optimizer`. Will be modified by the
        scheduler to overwrite the learning rate.
      scheduler_type: string, type of learning rate scheduler to use. If None,
        this method returns None.
      scheduler_divisor: int, required for WarmupAndExponentialDecayScheduler.
      scheduler_divide_every_n_epochs: int, required for
        WarmupAndExponentialDecayScheduler.
      num_steps_per_epoch: int, the number of steps that occur in each epoch.
        Required for WarmupAndExponentialDecayScheduler.
      summary_writer: Instance of `torch.utils.tensorboard.SummaryWriter` that
        will be passed into the scheduler to log learning rate during training.
    Raises:
      ValueError if the requested scheduler_type is unrecognized or if any
          required params are missing for the requested scheduler_type.
    """
    if not scheduler_type:
        return None

    if scheduler_type == 'WarmupAndExponentialDecayScheduler':
        if scheduler_divisor is None:
            raise ValueError('scheduler_divisor is required for '
                             'WarmupAndExponentialDecayScheduler.')
        if scheduler_divide_every_n_epochs is None:
            raise ValueError('scheduler_divide_every_n_epochs is required for '
                             'WarmupAndExponentialDecayScheduler.')
        if num_steps_per_epoch is None:
            raise ValueError('num_steps_per_epoch is required for '
                             'WarmupAndExponentialDecayScheduler.')
        return WarmupAndExponentialDecayScheduler(
            optimizer,
            num_steps_per_epoch,
            divide_every_n_epochs=scheduler_divide_every_n_epochs,
            divisor=scheduler_divisor,
            summary_writer=summary_writer)
    else:
        raise ValueError('Unknown scheduler_type: {}'.format(scheduler_type))
