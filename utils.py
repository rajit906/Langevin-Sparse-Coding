import numpy as np
import scipy.special
import scipy.stats
import torch
import torchvision
import matplotlib.pyplot as plt


def KLDivergence(p,q):
  p /= np.sum(p)
  q /= np.sum(q)

  vec = scipy.special.rel_entr(p, q)    
  kl_div = np.sum(vec)  
  return kl_div

def phi(s, s0):
    s[s < -s0] += s0
    s[s > s0] -= s0
    s[(s >= -s0) & (s <= s0)] = 0
    return s

def moments(S):
  mean = S.mean(axis = 0)
  variance = scipy.stats.tvar(S, axis = 0)
  skewness = scipy.stats.skew(S, axis = 0)
  kurtosis = scipy.stats.kurtosis(S, axis = 0)
  print("Mean:" + str(mean) + ", Sample Variance:" + str(variance) + ", Skewness:" + str(skewness) + ", Kurtosis:" + str(kurtosis))
  return mean, variance, skewness, kurtosis

def plot_marginal_distributions(S):
  n = S.shape[1]
  fig, axes = plt.subplots(ncols=n, figsize=(12, 4))
  for i in range(n):
    marginal = S[:,i]
    mean, variance, skewness, kurtosis = moments(marginal)
    print("Moments of S" + str(i) + ":" + " mean-" + str(mean) + ", variance-" + str(variance) + ", skewness-" + str(skewness) + ", kurtosis-" + str(kurtosis))
    axes[i].hist(S[:, i], fc='grey', bins=100, density=True)
    axes[i].set_xlabel('s' + str(i))
  plt.show()

def univariate_autocorrelation(data, lag = 10):
  size = 2 ** np.ceil(np.log2(2*len(data) - 1)).astype('int')
  var = np.var(data)
  ndata = data - np.mean(data)
  fft = np.fft.fft(ndata, size)
  pwr = np.abs(fft) ** 2
  acorr = np.fft.ifft(pwr).real / var / len(data)
  return acorr[0:lag]

def snr(vref, vcmp):
  dv = torch.dot(vref.flatten(), vref.flatten())
  vdiff = vref-vcmp
  rt = dv / torch.dot(vdiff.flatten(), vdiff.flatten())
  return (10.0 * torch.log10(rt)).item()

def plot_dictionary(dictionary, color=False, nrow=30, normalize=True,
                    scale_each=True, fig=None, ax=None, title="", size=8):

    n_features, n_basis = dictionary.shape
    nch = 1
    if color:
        nch = 3
    patch_size = int(np.sqrt(n_features//nch))
    D_imgs = dictionary.T.reshape([n_basis, patch_size, patch_size, nch]).permute([
        0, 3, 1, 2])  # swap channel dims for torch
    grid_img = torchvision.utils.make_grid(
        D_imgs, nrow=nrow, normalize=normalize, scale_each=scale_each).cpu()
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.clear()
    ax.set_title(title)
    ax.imshow(grid_img.permute(1, 2, 0))  # swap channel dims for matplotlib
    ax.set_axis_off()
    fig.set_size_inches(size, size)
    fig.canvas.draw()
    return fig, ax

def sample_random_patches(
    patch_size: int,
    num_patches: int,
    image: torch.Tensor,
):
    P = patch_size
    N = num_patches
    H, W = image.shape[-2:]

    h_start_idx = torch.randint(
        low=0,
        high=H - P + 1,
        size=(N,),
    )
    w_start_idx = torch.randint(
        low=0,
        high=W - P + 1,
        size=(N,),
    )

    h_patch_idxs, w_patch_idxs = torch.meshgrid(
        torch.arange(P),
        torch.arange(P),
        indexing='ij'
    )
    h_idxs = h_start_idx.reshape(N, 1, 1) + h_patch_idxs
    w_idxs = w_start_idx.reshape(N, 1, 1) + w_patch_idxs

    leading_idxs = [
        torch.randint(low=0, high=image.shape[d], size=(N, 1, 1))
        for d in range(image.dim() - 3)
    ]
    idxs = leading_idxs + [slice(None), h_idxs, w_idxs]
    patches = image[idxs]  # [N, P, P, C]
    return torch.permute(patches, (0, 3, 1, 2))