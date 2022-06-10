## MIMI: Mutual Information-Maximizing Interface

[MIMI](https://arxiv.org/abs/2205.12381) is an algorithm for training an interface to map user command signals to system actions through unsupervised human-in-the-loop reinforcement learning.

## Usage

1.  Clone `mimi` into your home directory `~`
2.  Download [data.zip](https://drive.google.com/file/d/1WEHJFkbBT3t1hu2Ici8o_k-_CxAUuSol/view?usp=sharing) and decompress it into `mimi/`
3.  Setup an Anaconda virtual environment with `conda create -n mimienv python=3.6`
4.  Install dependencies with `pip install -r requirements.txt` and `pip install pyglet==1.5.11`
5.  Replace `your_install_dir/gym/envs/box2d/lunar_lander.py` with `deps/box2d/lunar_lander.py`
6.  Install the `mimi` package with `python setup.py install`
7.  Jupyter notebooks in `mimi/notebooks` provide an entry-point to the code base, where you can
    play around with the environments and reproduce the figures from the paper.

## Citation

If you find this software useful in your work, we kindly request that you cite the following
[paper](https://arxiv.org/abs/2205.12381):

```
@article{mimi2022,
  title={First Contact: Unsupervised Human-Machine Co-Adaptation via Mutual Information Maximization},
  author={Reddy, Siddharth and Levine, Sergey and Dragan, Anca D.},
  journal={arXiv preprint arXiv:2205.12381},
  year={2022}
}
```

## Latent Space Exploration (Experimental)

Explore the latent space of a generative model of MNIST images using hand gestures

1.  Clone [this repo](https://github.com/YannDubs/disentangling-vae/commit/7b8285baa19d591cf34c652049884aca5d8acbca)
2.  Set `dvae_dir` in `mimi/utils.py`
3.  Download the [MNIST](https://github.com/lucastheis/deepbelief/blob/master/data/mnist.npz) dataset to `mimi/data/mnist/mnist.npz`
3.  Run `notebooks/mnist.ipynb`