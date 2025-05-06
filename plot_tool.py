import random
import string
import warnings
import json
from typing import Optional

import numpy as np
from matplotlib.colors import Colormap
from shap._explanation import Explanation
import matplotlib.pyplot as pl
from shap.plots import colors
from shap.utils._legacy import kmeans

def image(shap_values: Explanation or np.ndarray,
          pixel_values: Optional[np.ndarray] = None,
          labels: Optional[list or np.ndarray] = None,
          true_labels: Optional[list] = None,
          width: Optional[int] = 20,
          aspect: Optional[float] = 0.2,
          hspace: Optional[float] = 0.2,
          labelpad: Optional[float] = None,
          cmap: Optional[str or Colormap] = colors.red_transparent_blue,
        #   cmap: Optional[str or Colormap] = colors.red_blue_no_bounds,
          show: Optional[bool] = True,
          save_path: Optional[str] = None,
          choose: Optional[list] = None,
          color:Optional[bool] = True,
          custom_bar:Optional[bool] = True,
          shapley_mask:Optional[list] = None):
    
    """ Plots SHAP values for image inputs.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being explained.

    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image. It should be the same
        shape as each array in the shap_values list of arrays.

    labels : list or np.ndarray
        List or np.ndarray (# samples x top_k classes) of names for each of the model outputs that are being explained.

    true_labels: list
        List of a true image labels to plot

    width : float
        The width of the produced matplotlib plot.

    labelpad : float
        How much padding to use around the model output labels.

    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """

    if shapley_mask is not None:
        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i]*shapley_mask[i]
        
    lst = []
    if choose is not None:
        for i in choose:
            lst.append(shap_values[i])
        shap_values = lst
    label_kwargs = {} if labelpad is None else {'pad': labelpad}
# plot our explanations
    x = pixel_values
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = pl.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size) 
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure we have a 2D array for grayscale
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])

        # if x_curr.max() > 1:
        #     x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                    0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
            x_curr_disp = x_curr
        elif len(x_curr.shape) == 3:
            x_curr_gray = x_curr.mean(2)

            # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
            flat_vals = x_curr.reshape([x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]).T
            flat_vals = (flat_vals.T - flat_vals.mean(1)).T
            means = kmeans(flat_vals, 3, round_values=False).data.T.reshape([x_curr.shape[0], x_curr.shape[1], 3])
            x_curr_disp = (means - np.percentile(means, 0.5, (0, 1))) / (
                    np.percentile(means, 99.5, (0, 1)) - np.percentile(means, 1, (0, 1)))
            x_curr_disp[x_curr_disp > 1] = 1
            x_curr_disp[x_curr_disp < 0] = 0
        else:
            x_curr_gray = x_curr
            x_curr_disp = x_curr

        axes[row, 0].imshow(x_curr_disp, cmap=pl.get_cmap('gray')) 
        if true_labels:
            axes[row, 0].set_title(true_labels[row], **label_kwargs)
        axes[row, 0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
        for i in range(len(shap_values)):
            if labels is not None:
                axes[row, i + 1].set_title(labels[row, i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row, i + 1].imshow(x_curr_gray, cmap=pl.get_cmap('gray'), alpha=0.6,
                                    extent=(-1, sv.shape[1], sv.shape[0], -1))      
            if custom_bar:
                sorted_sv = np.sort(sv.flatten())
                result = sorted_sv[int(sorted_sv.size*0.4)] 
                # im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=2*result-max_val, vmax=max_val) 
                im = axes[row, i + 1].imshow(sv, cmap='plasma', alpha=0.7, vmin=0.0, vmax=0.3)
            else:
                # im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
                im = axes[row, i + 1].imshow(sv, cmap='plasma', alpha=0.7, vmin=0.0, vmax=0.3)

            axes[row, i + 1].axis('off')
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    if color == True:
        cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
                          aspect=fig_size[0] / aspect)
        cb.outline.set_visible(False)
    if show:
        pl.show()
    if save_path is not None:
        pl.savefig(save_path)
        pl.close()
