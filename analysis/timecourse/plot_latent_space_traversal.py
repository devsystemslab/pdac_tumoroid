from pathlib import Path

import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from scipy.stats import spearmanr
from skimage import io
from skimage.measure import regionprops
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from tqdm import tqdm

from phenocoder.phenocode import load_phenocoder
from phenocoder.utils import load_plate


def get_patch(
    label, df_plates, patch_size=(100, 100), plot=False, calculate_regionprops=False
):
    label_row = df_plates[df_plates["label"] == label]
    z_init = int(label_row["z_init"].iloc[0])
    z = z_init + 1
    x = label_row["centroid-0"].item()
    y = label_row["centroid-1"].item()
    well = label_row["well"].item()
    plate = label_row["plate"].item()

    seg_id = int(label.split("_")[0])

    df_summary_plate = pd.read_csv(
        f"/pstore/data/ihb-tumoroid/data/processed/timecourse/{plate}/{plate}-01/features/nuclei/df_summary_TIF_OVR_BG.csv"
    )

    summary_row = df_summary_plate[
        (df_summary_plate["well"] == well)
        & (df_summary_plate["z_stack"] == z)
        & (df_summary_plate["channel"] == 1)
    ]
    dir_images = summary_row["dir_images"].item()
    img_file = summary_row["file"].item()

    seg_path = Path(dir_screen, plate, f"{plate}-01", "SEG_TIF_OVR_BG", img_file)
    img_seg = io.imread(seg_path)

    img = io.imread(Path(dir_screen, plate, f"{plate}-01", "TIF_OVR_BG", img_file))

    cx, cy = int(round(x)), int(round(y))
    x0 = cx - patch_size[0] // 2
    y0 = cy - patch_size[1] // 2
    x1 = x0 + patch_size[0]
    y1 = y0 + patch_size[1]

    patch = img[x0:x1, y0:y1]

    patch_seg = img_seg[x0:x1, y0:y1]
    patch_seg[patch_seg != seg_id] = 0
    props = regionprops(patch_seg)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(img, cmap="viridis")
        axes[0].add_patch(
            plt.Rectangle(
                (y0, x0),
                patch_size[1],
                patch_size[0],
                edgecolor="red",
                facecolor="none",
                linewidth=1.5,
            )
        )
        axes[0].set_title("Full image — channel 1 (DAPI)")

        axes[1].imshow(patch, cmap="viridis")
        im = axes[2].imshow(patch_seg, cmap="viridis")
        fig.colorbar(im, ax=axes[2])

        if calculate_regionprops:
            prop = props[0]
            y_c, x_c = prop.centroid  # row, col

            length = prop.major_axis_length / 2

            dx = length * np.sin(prop.orientation)
            dy = length * np.cos(prop.orientation)

            axes[2].plot([x_c - dx, x_c + dx], [y_c - dy, y_c + dy], "r-", linewidth=2)

        plt.tight_layout()
        plt.show()

    return patch, z, plate, patch_seg, props


def calc_latent_morphology_correlation(
    adata,
    df_plates=None,
    adata_features=None,
    morphology_cols=None,
    use_pca=False,
    n_components=10,
):
    """
    Either df_plates (DataFrame) or adata_features (AnnData) must be provided.
    If adata_features is given, its .X columns are used as features.
    """
    X = np.array(adata.X)
    labels = adata.obs.index.tolist()

    if use_pca:
        pca = PCA(n_components=n_components)
        Z = pca.fit_transform(X)
        var = pca.explained_variance_ratio_
        dim_labels = [f"PC{i + 1} ({var[i] * 100:.1f}%)" for i in range(n_components)]
    else:
        Z = X
        dim_labels = [f"Z{i + 1}" for i in range(X.shape[1])]

    df_Z = pd.DataFrame(Z, index=labels, columns=dim_labels)

    # --- build feature dataframe from either source ---
    if adata_features is not None:
        feat_cols = (
            morphology_cols
            if morphology_cols is not None
            else adata_features.var.index.tolist()
        )
        df_morph = pd.DataFrame(
            np.array(adata_features.X),
            index=adata_features.obs.index,
            columns=adata_features.var.index,
        )[feat_cols]
    elif df_plates is not None:
        df_morph = df_plates.set_index("label")
        df_morph = df_morph[morphology_cols]
    else:
        raise ValueError("Either df_plates or adata_features must be provided.")

    # --- align ---
    shared = df_Z.index.intersection(df_morph.index)
    df_Z = df_Z.loc[shared]
    df_morph = df_morph.loc[shared]

    # --- Spearman correlations ---
    n_dims = len(dim_labels)
    n_feats = len(df_morph.columns)
    corr_matrix = np.zeros((n_dims, n_feats))
    pval_matrix = np.zeros_like(corr_matrix)

    for i, dim_col in enumerate(dim_labels):
        for j, feat in enumerate(df_morph.columns):
            print(f"Calc {dim_col} {feat}")
            valid = df_morph[feat].notna()
            r, p = spearmanr(df_Z.loc[valid, dim_col], df_morph.loc[valid, feat])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p

    df_corr = pd.DataFrame(corr_matrix, index=dim_labels, columns=df_morph.columns)
    df_pval = pd.DataFrame(pval_matrix, index=dim_labels, columns=df_morph.columns)

    return df_corr, df_pval


def plot_latent_morphology_correlation(
    df_corr, use_pca=False, plot_only_top=None, sort_corr_order=False, figsize=(10, 6)
):
    if sort_corr_order:
        row_order = df_corr.abs().max(axis=1).sort_values(ascending=False).index
        df_corr = df_corr.loc[row_order]

    if plot_only_top is not None:
        df_corr = df_corr.iloc[:plot_only_top]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_corr,
        ax=ax,
        cmap="RdBu",
        center=0,
        linewidths=0.3,
        linecolor="lightgray",
        cbar_kws={"label": "Spearman r", "shrink": 0.6},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig


def plot_decoder_traversal_from_cell(
    model,
    oh_enc,
    adata,
    cell_label,
    dim_idx,
    n_samples=9,
    n_std=2.0,
    figsize=(18, 2),
    patch_size=(100, 100),
):
    dim_idx = dim_idx - 1

    # get the original patch for reference
    patch, z, plate, patch_seg, props = get_patch(
        cell_label, df_plates, patch_size=patch_size, calculate_regionprops=False
    )

    # get the cell's latent vector as anchor
    z_anchor = np.array(adata[cell_label].X).flatten()  # (64,)

    # range to traverse around the cell's own value for dim_idx
    all_scores = np.array(adata.X)[:, dim_idx]
    std = np.std(all_scores)
    center = z_anchor[dim_idx]
    traversal_values = np.linspace(
        center - n_std * std, center + n_std * std, n_samples
    )

    # condition
    condition = [f"{plate}-01", z]
    cond_encoded = oh_enc.transform([condition]).toarray()

    # decode unmodified reconstruction
    z_anchor_tensor = tf.convert_to_tensor(z_anchor[np.newaxis], dtype=tf.float32)
    cond_tensor_single = tf.convert_to_tensor(cond_encoded, dtype=tf.float32)
    recon_img = model.decoder(
        [z_anchor_tensor, cond_tensor_single]
    ).numpy()  # (1, H, W, C)

    # build z batch for traversal
    z_batch = np.tile(z_anchor, (n_samples, 1))
    z_batch[:, dim_idx] = traversal_values

    # decode traversal batch
    z_tensor = tf.convert_to_tensor(z_batch, dtype=tf.float32)
    cond_tensor = tf.convert_to_tensor(
        np.tile(cond_encoded, (n_samples, 1)), dtype=tf.float32
    )
    imgs = model.decoder([z_tensor, cond_tensor]).numpy()

    # plot: original | reconstruction | traversal images
    fig, axes = plt.subplots(
        1, n_samples + 2, figsize=figsize, gridspec_kw={"wspace": 0.05}
    )

    # original patch
    axes[0].imshow(patch, cmap="gray")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].text(
        patch.shape[1] / 2,
        4,
        f"original",
        color="white",
        fontsize=7,
        ha="center",
        va="top",
    )
    axes[0].text(
        patch.shape[1] / 2,
        patch.shape[0] - 4,
        f"Z{dim_idx + 1}={center:.2f}",
        color="white",
        fontsize=7,
        ha="center",
        va="bottom",
    )

    # unmodified reconstruction
    axes[1].imshow(recon_img[0, :, :, 0], cmap="gray")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].text(
        recon_img.shape[2] / 2,
        4,
        "reconstruction",
        color="white",
        fontsize=7,
        ha="center",
        va="top",
    )
    axes[1].text(
        recon_img.shape[2] / 2,
        recon_img.shape[1] - 4,
        f"{center:.2f}",
        color="white",
        fontsize=7,
        ha="center",
        va="bottom",
    )

    # traversal images
    for i, ax in enumerate(axes[2:]):
        ax.imshow(imgs[i, :, :, 0], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            imgs.shape[2] / 2,
            imgs.shape[1] - 4,
            f"{traversal_values[i]:.2f}",
            color="white",
            fontsize=7,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    return fig


def add_neighbor_count(df_plates, radius=50):
    """
    For each cell, count the number of neighbors within `radius` pixels
    in the same well, plate and z_init. Adds a 'neighbor_count' column.
    """
    df_plates = df_plates.copy()
    df_plates[f"neighbor_count_{radius}"] = 0

    groups = df_plates.groupby(["well", "plate", "z_init"])

    for (well, plate, z_init), group_idx in groups.groups.items():
        group = df_plates.loc[group_idx]
        coords = group[["centroid-0", "centroid-1"]].values

        if len(coords) < 2:
            continue

        tree = BallTree(coords)
        counts = tree.query_radius(coords, r=radius, count_only=True)

        # subtract 1 to exclude the cell itself
        df_plates.loc[group_idx, f"neighbor_count_{radius}"] = counts - 1

    return df_plates


def orientation_embedding_correlation(
    adata,
    df_plates,
    n_sample=1000,
    patch_size=(128, 128),
):
    # --- Sample labels ---
    rng = np.random.default_rng()
    all_labels = adata.obs_names.tolist()
    sampled_labels = rng.choice(
        all_labels, size=min(n_sample, len(all_labels)), replace=False
    )

    # --- Get orientations ---
    orientations = []
    valid_labels = []

    for label in tqdm(sampled_labels, desc="Extracting patches"):
        try:
            _, _, _, patch_seg, props = get_patch(
                label,
                df_plates,
                patch_size=patch_size,
                plot=False,
                calculate_regionprops=True,
            )
            if len(props) == 0:
                continue
            orientations.append(props[0].orientation)
            valid_labels.append(label)
        except Exception as e:
            print(f"Skipping {label}: {e}")
            continue

    orientations = np.array(orientations)
    sin_vals = np.sin(2 * orientations)
    cos_vals = np.cos(2 * orientations)

    # --- Subset adata to valid labels ---
    adata_sub = adata[valid_labels]
    X = adata_sub.X
    if hasattr(X, "toarray"):  # handle sparse
        X = X.toarray()

    # --- Correlate against each embedding dimension ---
    results = []
    for dim in tqdm(range(X.shape[1]), desc="Computing correlations"):
        emb = X[:, dim]
        r_sin, p_sin = spearmanr(sin_vals, emb)
        r_cos, p_cos = spearmanr(cos_vals, emb)
        r_combined = np.sqrt(r_sin**2 + r_cos**2)
        results.append(
            {
                "dim": dim,
                "r_sin": r_sin,
                "p_sin": p_sin,
                "r_cos": r_cos,
                "p_cos": p_cos,
                "r_combined": r_combined,
            }
        )

    results_df = pd.DataFrame(results).sort_values("r_combined", ascending=False)
    return results_df, orientations


def plot_orientation_correlation_results(results_df):
    top20 = results_df.nlargest(20, "r_combined").sort_values(
        "r_combined", ascending=False
    )

    heatmap_data = top20[["r_sin", "r_cos", "r_combined"]].set_index(
        top20["dim"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(heatmap_data.values, cmap="coolwarm", aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["r_sin", "r_cos", "r_combined"])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_title("Top 20 dims by r_combined")

    fig.colorbar(im, ax=ax, label="Spearman r")
    plt.tight_layout()
    plt.show()


screen = "timecourse"
file = "/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/configs/params.yaml"

with open(file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params[screen]

dir_screen = params["dir_screen"]
dir_results = Path(params["dir_screen"], "anndata")
plates = params["plates"]

mdata = mu.read_h5mu(Path(dir_results, "mdata_registered_imputed_mlp_normalized.h5mu"))

df_plates = pd.concat(
    [
        pd.read_csv(
            f"{dir_screen}/{plate}/plate_information.csv", dtype={plate: str}
        ).assign(plate=plate)
        for plate in plates
    ]
)

plates = params["plates"]
df_plates = []
for plate in plates:
    df_plate = load_plate(
        plate,
        params["input_type"],
        params["dir_screen"],
        params["registered"],
        z_step=10,
        plate_id=f"{plate}-01",
    )
    df_plates.append(df_plate)
df_plates = pd.concat(df_plates)
df_plates["label"] = (
    df_plates["label"].astype(str) + "_" + df_plates["well"] + "_" + df_plates["plate"]
)

# df_plates = pd.read_pickle('df_plates.pkl')

adata = mdata.mod["phenocoder_msg"]

radius = 100
df_plates = add_neighbor_count(df_plates, radius=radius)

# calc and plot correlation latent dims and morph features
df_corr, df_pval = calc_latent_morphology_correlation(
    adata,
    df_plates,
    morphology_cols=[
        "area",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "ch_01_nuclei",
        "ch_01_neighbors",
        f"neighbor_count_100",
    ],
    n_components=8,
    use_pca=False,
)

fig = plot_latent_morphology_correlation(
    df_corr, plot_only_top=20, figsize=(4, 8), sort_corr_order=True
)

# calc and plot correlation latent dims and imputed marker values
df_corr_imputed, df_pval_imputed = calc_latent_morphology_correlation(
    adata,
    adata_features=mdata.mod["phenocoder_msg_neighbors_imputed"],
    n_components=8,
    use_pca=False,
)

fig = plot_latent_morphology_correlation(
    df_corr_imputed,
    df_pval_imputed,
    plot_only_top=20,
    figsize=(12, 8),
    sort_corr_order=True,
)

dir_models = params["phenocoder"]["dir_models"]
model_dict = params["phenocoder"]["models"]
model, oh_enc, config = load_phenocoder(Path(dir_models, model_dict["source"]["file"]))

patch_size = (128, 128)

# labels = ['33928_P04_003', '41186_B02_002', '49035_L02_004'] #major axis, latent 46, std 4, encodes center/outside when its a crowded, round cell
# labels = ['98_E04_004', '40895_F04_004', '37366_I05_004'] #major axis, latent 46, std 3, encodes major axis when its a long cell
# labels = ['11921_J03_002', '153_B02_001', '262_N05_001'] # minor axis, latent 50, std 5
# labels = ['98_E04_004', '22896_B05_003', '3444_N01_005'] # rotation, latent 34, std 10
# labels = ['10562_I03_003', '35827_P04_002', '1734_B01_005'] # nuclei/neighbor, latent 4, std 5
labels = adata.obs_names[
    np.random.choice(len(adata.obs_names), 10, replace=False)
].tolist()
for label in labels:
    dim_idx = 46
    patch, z, plate, patch_seg, props = get_patch(
        label, df_plates, patch_size=patch_size, plot=False, calculate_regionprops=True
    )
    fig = plot_decoder_traversal_from_cell(
        model,
        oh_enc,
        adata,
        cell_label=label,
        dim_idx=dim_idx,
        n_samples=10,
        n_std=4.0,
        patch_size=patch_size,
    )
    # fig.savefig(f'plots_new/latent_{dim_idx}_{label}.pdf', bbox_inches='tight', dpi=72)

results_df, orientations = orientation_embedding_correlation(
    adata, df_plates, n_sample=1000, patch_size=patch_size
)

# df_corr = pd.read_pickle('df_corr.pkl')
df_corr.index = df_corr.index.str.lstrip("Z").astype(int)
results_df.index = results_df.index - 1
df_merged = df_corr.join(results_df)
df_merged = df_merged[
    [
        "area",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "ch_01_nuclei",
        "ch_01_neighbors",
        "neighbor_count_100",
        "r_sin",
        "r_cos",
        "r_combined",
    ]
]
fig = plot_latent_morphology_correlation(
    df_merged, plot_only_top=20, figsize=(4, 8), sort_corr_order=True
)


sampled_labels = adata.obs_names[
    np.random.choice(len(adata.obs_names), 1000, replace=False)
].tolist()
X_sub = adata[sampled_labels].X
if hasattr(X_sub, "toarray"):
    X_sub = X_sub.toarray()

# Normalise each dimension to unit length
norms = np.linalg.norm(X_sub, axis=0, keepdims=True)
X_norm = X_sub / norms

# Cosine similarity matrix between all pairs of dimensions
cosine_sim = X_norm.T @ X_norm  # shape: (n_dims, n_dims)

sns.clustermap(cosine_sim, cmap="coolwarm", vmin=-1, vmax=1, figsize=(20, 20))
plt.show()
