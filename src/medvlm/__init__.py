"""Image-reliance audit of medical VLMs."""
import os

# transformers imports TensorFlow/Flax when they happen to be installed (slow start, noisy logs);
# this project is PyTorch-only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

__version__ = "0.1.0"
