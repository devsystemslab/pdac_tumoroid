import os
import pickle
from pathlib import Path

import anndata as ad
import marsilea as ma
import marsilea.plotter as mp
import matplotlib.pyplot as plt
import muon as mu
import networkx as nx
import numpy as np
import pandas as pd
import pynndescent
import scanpy as sc
import seaborn as sns
import yaml
from matplotlib import colors
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from whole_mount_tumoroid.phenocoder.cluster import run_clustering
from whole_mount_tumoroid.phenocoder.utils import load_plate

screen = "timecourse"
file = "whole_mount_tumoroid/configs/params.yaml"

with open(file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params[screen]

dir_results = Path(params["dir_screen"], "anndata")

mdata = mu.read_h5mu(Path(dir_results, "mdata_registered_imputed.h5mu"))

mdata.mod[f"phenocoder"] = run_clustering(
    mdata["phenocoder"],
    subsampling=True,
    frac=0.1,
    n_comps=params["phenocoder"]["n_comps_pca"]["phenocoder"],
    resolution=params["phenocoder"]["cluster_res"]["phenocoder"],
    harmony=False,
    use_gpu=False,
)

mdata.mod["phenocoder_msg"] = run_clustering(
    mdata["phenocoder_msg"],
    subsampling=True,
    frac=0.1,
    n_comps=params["phenocoder"]["n_comps_pca"]["phenocoder"],
    resolution=params["phenocoder"]["cluster_res"]["phenocoder"],
    harmony=False,
    use_gpu=False,
)

mdata.write_h5mu(Path(dir_results, "mdata_registered_imputed_newclusters.h5mu"))
