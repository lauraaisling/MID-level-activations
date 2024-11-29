import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image # to visualize images

# Call `set_seed` function in the exercises to ensure reproducibility.
def set_seed(seed=None, seed_torch=True):
  """
  Handles variability by controlling sources of randomness
  through set seed values

  Args:
    seed: Integer
      Set the seed value to given integer.
      If no seed, set seed value to random integer in the range 2^32
    seed_torch: Bool
      Seeds the random number generator for all devices to
      offer some guarantees on reproducibility

  Returns:
    Nothing
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


# Inform the user if the notebook uses GPU or CPU.
def set_device():
  """
  Set the device. CUDA if available, CPU otherwise

  Args:
    None

  Returns:
    Nothing
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: CPU runtime")
  else:
    print(f"{device} device is enabled.")

  return device


  # Function to plot a histogram of RSM values: `plot_rsm_histogram(rsms, colors)`
def plot_rsm_histogram(rsms, colors, labels=None, nbins=100):
  """
  Function to plot histogram based on Representational Similarity Matrices

  Args:
    rsms: List
      List of values within RSM
    colors: List
      List of colors for histogram
    labels: List
      List of RSM Labels
    nbins: Integer
      Specifies number of histogram bins

  Returns:
    Nothing
  """
  fig, ax = plt.subplots(1)
  ax.set_title("Histogram of RSM values", y=1.05)

  min_val = np.min([np.nanmin(rsm) for rsm in rsms])
  max_val = np.max([np.nanmax(rsm) for rsm in rsms])

  bins = np.linspace(min_val, max_val, nbins+1)

  if labels is None:
    labels = [labels] * len(rsms)
  elif len(labels) != len(rsms):
    raise ValueError("If providing labels, must provide as many as RSMs.")

  if len(rsms) != len(colors):
    raise ValueError("Must provide as many colors as RSMs.")

  for r, rsm in enumerate(rsms):
    ax.hist(
        rsm.reshape(-1), bins, density=True, alpha=0.4,
        color=colors[r], label=labels[r]
        )
  ax.axvline(x=0, ls="dashed", alpha=0.6, color="k")
  ax.set_ylabel("Density")
  ax.set_xlabel("Similarity values")
  ax.legend()
  plt.show()


# Function to set test custom torch RSM function: `test_custom_torch_RSM_fct()`
def test_custom_torch_RSM_fct(custom_torch_RSM_fct):
  """
  Function to set test implementation of custom_torch_RSM_fct

  Args:
    custom_torch_RSM_fct: f_name
      Function to test

  Returns:
    Nothing
  """
  rand_feats = torch.rand(100, 1000)
  RSM_custom = custom_torch_RSM_fct(rand_feats)
  RSM_ground_truth = data.calculate_torch_RSM(rand_feats)

  if torch.allclose(RSM_custom, RSM_ground_truth, equal_nan=True):
    print("custom_torch_RSM_fct() is correctly implemented.")
  else:
    print("custom_torch_RSM_fct() is NOT correctly implemented.")


