"""Setup script for KS-PlaNet (KS + Kolmogorov adaptations).

Install one of three configurations:

  pip install -e ".[local]"     -- laptop with old TF 1.13 GPU (legacy path)
  pip install -e ".[cpu]"       -- laptop CPU-only (works on any machine)
  pip install -e ".[container]" -- inside the NGC TF1 container
                                  (TF 1.15.4 + CUDA 11.1, runs on RTX 30xx / A100)

  pip install -e .              -- platform-agnostic only (container provides TF)
"""

from __future__ import absolute_import, division, print_function
import setuptools


# ── Platform-agnostic deps (install everywhere, no TF) ──────────────────────
COMMON = [
    'gym>=0.17.0,<=0.21.0',           # 0.22+ changes step() signature
    'dm_control',                      # required by PlaNet import chain
    'ruamel.yaml>=0.15.0,<0.16.0',    # 0.16 has incompatible API
]


# ── Inside the NGC TF1 container (nvcr.io/nvidia/tensorflow:20.10-tf1-py3) ──
# Container provides: TF 1.15.4, CUDA 11.1, Python 3.6, numpy, scipy, matplotlib
# We only add what's missing.
CONTAINER = [
    'tensorflow_probability==0.8.0',   # matches TF 1.15
    'scikit-image>=0.17.0',            # not in NGC image
]


# ── Local laptop with old GPU (TF 1.13, CUDA 10, Python 3.7) ────────────────
# Will NOT work on Ampere GPUs (RTX 30xx, A100). Use [cpu] or [container] instead.
LOCAL_GPU = [
    'tensorflow-gpu==1.13.1',
    'tensorflow_probability==0.6.0',
    'numpy>=1.16.0,<1.20.0',           # 1.20+ removes np.bool used by TF1
    'scipy>=1.4.0,<1.7.0',
    'scikit-image>=0.17.0,<0.19.0',
    'matplotlib>=3.3.0,<3.6.0',
    'Pillow>=8.0.0,<10.0.0',
    'protobuf>=3.6.1,<3.20.0',
]


# ── Local CPU-only (works anywhere) ────────────────────────────────────────
LOCAL_CPU = [
    'tensorflow==1.13.1',
    'tensorflow_probability==0.6.0',
    'numpy>=1.16.0,<1.20.0',
    'scipy>=1.4.0,<1.7.0',
    'scikit-image>=0.17.0,<0.19.0',
    'matplotlib>=3.3.0,<3.6.0',
    'Pillow>=8.0.0,<10.0.0',
    'protobuf>=3.6.1,<3.20.0',
]


setuptools.setup(
    name='ks-planetrl',
    version='1.0.0',
    description=('PlaNet adapted for Kuramoto-Sivashinsky and Kolmogorov flow '
                 'world-model learning.'),
    license='Apache 2.0',
    url='https://github.com/google-research/planet',
    python_requires='>=3.6, <3.11',
    install_requires=COMMON,
    extras_require={
        'container': CONTAINER,
        'local':     LOCAL_GPU,
        'cpu':       LOCAL_CPU,
        'dev':       ['pytest>=6.0', 'pytest-cov'],
    },
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
    ],
)
