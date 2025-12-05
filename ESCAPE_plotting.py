import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt

from ESCAPE_evaluation import smooth_array

def adjust_color_brightness(hex_color, factor=1.0):
    rgb = mcolors.to_rgb(hex_color)
    adjusted_rgb = tuple(min(1, max(0, c * factor)) for c in rgb)
    return mcolors.to_hex(adjusted_rgb)


def plot_dca(dca_df, ax, net_benefit_model, model_name="model", pt_thresholds=np.arange(0.01, 1.0, 0.01), show_thresholds=[0.05, 0.15], smooth_fraction=0, color="blue"):
    if 1 > smooth_fraction > 0:
        pt_thresholds, net_benefit_model = smooth_array(pt_thresholds, net_benefit_model, smooth_frac=smooth_fraction)      
    ax.plot(pt_thresholds, net_benefit_model, label=model_name, color=color)
    for pt in show_thresholds:
        ax.axvline(x=pt, linestyle=':', color='green', alpha=1)
        #ax.text(pt + 0.01, max(net_benefit_model)*0.9, pt, rotation=90, color='green', fontsize=8, alpha=1)