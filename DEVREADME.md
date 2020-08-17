# Tips/Information on basic functionality of codebase


## one hot labels that appear in losses

one_hot_labels is 2*k wide, as in
`/gan/tensorflow_gan/python/estimator/gan_estimator.py:_make_gan_model` and
`/gan/tensorflow_gan/python/estimator/tpu_gan_estimator.py:_make_gan_model_fns`
and more importantly in `/gan/tensorflow_gan/python/train.py:acgan_model`

And k = num classes, except for a kplusonegan, k = num_classes + 1

## to add a new loss

- write fn in `/gan/tensorflow_gan/python/losses/losses_impl.py`, add to `__all__` there
- add in `/gan/tensorflow_gan/python/losses/tuple_losses.py` like so:
```
kplusonegan_nll_discriminator_loss = args_to_gan_model(
    tfgan_losses.kplusonegan_nll_discriminator_loss)
```

If its a --generator_loss_fn.

- add it to the --generator_loss_fn enums in `/gan/tensorflow_gan/examples/self_attention_estimator/train_experiment_main.py`
- add it to `gen_losses hash` in `get_tpu_estimator` and `get_gpu_estimator` in `/gan/tensorflow_gan/examples/self_attention_estimator/estimator_lib.py`

Otherwise:

- add a flag in `/gan/tensorflow_gan/examples/self_attention_estimator/train_experiment_main.py`
- add in `/gan/tensorflow_gan/python/train.py` in blocks related to:
    + args of gan_loss fn
    + validate args
    + verify config
    + using the loss



## Flags info

for K+1 gan, need --critic_type=kplusone_fm so that network will return (features, K+1 logits) tuple

by setting --generator_loss_fn, main loss for discriminator becomes nothing,
so add in the other loss through auxiliary