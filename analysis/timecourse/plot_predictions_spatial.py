import io as io_builtin
from pathlib import Path

import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from skimage import io
from skimage.util import montage
from sklearn.linear_model import LinearRegression

from phenocoder.utils import load_plate


def regress_out_z(
    df: pd.DataFrame,
    intensity_col: str = "intensity_value",
    stain_col: str = "stain",
    z_col: str = "z",
) -> pd.DataFrame:
    df = df.copy()

    def _regress(group):
        X = group[[z_col]].values  # shape (n, 1)
        y = group[intensity_col].values  # shape (n,)
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        group = group.copy()
        group[intensity_col] = residuals
        return group

    df = df.groupby(stain_col, group_keys=False).apply(_regress)
    return df


def scale_per_stain(
    df: pd.DataFrame,
    intensity_col: str = "intensity_value",
    stain_col: str = "stain",
    zero_center: bool = True,
    max_value=None,
) -> pd.DataFrame:
    df = df.copy()
    scaled = df.groupby(stain_col)[intensity_col].transform(
        lambda x: (
            (x - x.mean()) / (x.std(ddof=1) + 1e-10)
            if zero_center
            else x / (x.std(ddof=1) + 1e-10)
        )
    )
    if max_value is not None:
        scaled = scaled.clip(-max_value, max_value)
    df[intensity_col] = scaled
    return df


def data_setup(adata, df_stain_layout, df, imputation, stain_dict):
    obs = adata.obs.copy()
    obs = obs.reset_index().merge(
        df_stain_layout, how="left", on=["well_id", "plate_id"]
    )
    obs = obs.merge(
        df[
            [
                "label",
                f"ch_02_{imputation}",
                f"ch_03_{imputation}",
                f"ch_04_{imputation}",
            ]
        ],
        how="left",
        on=["label"],
    )

    obs_df = pd.melt(
        obs,
        id_vars=["label", "staining_set", "z"],
        value_vars=[
            f"ch_02_{imputation}",
            f"ch_03_{imputation}",
            f"ch_04_{imputation}",
        ],
        var_name="channel",
        value_name="intensity_value",
    )
    obs_df["channel"] = obs_df["channel"].str.split("_").str[1].astype(int)
    obs_df["stain"] = obs_df.apply(
        lambda row: stain_dict.get(str(row["staining_set"]), {}).get(row["channel"]),
        axis=1,
    )
    obs_df["stain"] = pd.Categorical(obs_df["stain"])
    obs_df["stain_id"] = obs_df["stain"].cat.codes

    obs_df = regress_out_z(obs_df)
    obs_df = scale_per_stain(obs_df)
    return obs_df


def plot_organoid_feature(
    id,
    adata,
    feature,
    project_3d=False,
    edgecolors=None,
    add_legend=False,
    cut_open=False,
    bg_color="black",
    legend_color="white",
    obs_df=None,
    plot_obs_df=False,
):
    """
    Plot organoid
    :param id:
    :param adata:
    :param project_3d:
    :param edgecolors:
    :param add_legend:
    :return:
    """
    adata = adata.copy()

    # Reset index to make it accessible as a column
    adata.obs = adata.obs.reset_index(drop=False)

    # Create id column if it doesn't exist
    if "id" not in adata.obs.columns:
        adata.obs["id"] = (
            adata.obs["well_id"].astype(str) + "_" + adata.obs["plate_id"].astype(str)
        )

    # Filter by id
    mask = adata.obs["id"] == id

    # Apply cut_open filter if needed
    if cut_open:
        center_centroid_0 = 3814 / 2
        center_centroid_1 = 3814 / 2
        mask = mask & ~(
            (adata.obs["centroid-0"] < center_centroid_0)
            & (adata.obs["centroid-1"] > center_centroid_1)
        )

    # Apply mask once and create clean copy (not a view)
    adata = adata[mask].copy()

    if adata.shape[0] == 0:
        print(f"No data for id {id}")
        return None

    # merge obs_df
    # filter obs_df for feature
    if obs_df is not None:
        # select columns stain, label, intensity value
        obs_df = obs_df[["stain", "label", "intensity_value"]]
        obs_df = obs_df[obs_df["stain"] == feature]
        obs_df = obs_df[obs_df["label"].isin(adata.obs["label"])]
        adata.obs = adata.obs.merge(obs_df, on="label")

    # Randomize order
    np.random.seed(0)
    random_indices = np.random.permutation(adata.shape[0])
    adata = adata[random_indices].copy()

    # Calculate quantile-based vmin/vmax
    if plot_obs_df:
        feature_data = adata.obs["intensity_value"]
    else:
        feature_data = adata.obs_vector(feature)
    # z-score feature data
    feature_data = (feature_data - feature_data.mean()) / feature_data.std()
    vmin = np.percentile(feature_data, 1)
    vmax = np.percentile(feature_data, 99)

    # scale feature data
    x_lim, y_lim = (0, 3814), (0, 3814)
    fig = plt.figure(figsize=(5, 5))
    # set background to black
    fig.patch.set_facecolor(bg_color)
    if project_3d:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_aspect("equal", "box")
        ax.scatter(
            adata.obs["centroid-1"],
            adata.obs["centroid-0"],
            adata.obs["z"],
            c=feature_data,
            s=20,
            alpha=0.75,
            edgecolors=edgecolors,
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,
        )
        # no axis labels
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.get_zaxis().set_ticklabels([])
        ax.set_xlim(xmin=x_lim[0], xmax=x_lim[1])
        ax.set_ylim(ymin=y_lim[0], ymax=y_lim[1])
        ax.set_facecolor(bg_color)
        if cut_open:
            # add line on x-y plane which show the cut off quadrant:
            ax.plot(
                [center_centroid_1, center_centroid_1],
                [0, 3814],
                color="black",
                linestyle="--",
                linewidth=1,
            )
            ax.plot(
                [0, 3814],
                [center_centroid_0, center_centroid_0],
                color="black",
                linestyle="--",
                linewidth=1,
            )
            # center y vertical line into z direction
            ax.plot(
                [center_centroid_1, center_centroid_1],
                [center_centroid_0, center_centroid_0],
                [0, adata.obs["z"].max()],
                color="black",
                linestyle="--",
                linewidth=1,
                zorder=10,
            )
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal", "box")
        ax.scatter(
            adata.obs["centroid-1"],
            adata.obs["centroid-0"],
            c=feature_data,
            s=20,
            edgecolors=edgecolors,
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,
        )
        # set axis limits
        ax.set_xlim(xmin=x_lim[0], xmax=x_lim[1])
        ax.set_ylim(ymin=y_lim[0], ymax=y_lim[1])
        # reverse y-axis
        ax.yaxis.set_inverted(True)
        # no axis labels
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        # no axis ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        # no frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # black background
        ax.set_facecolor(bg_color)
    if add_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1, 1),
            facecolor=legend_color,
            edgecolor=legend_color,
        )
    # tight layout
    plt.tight_layout()
    # reduce white margin around plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    # Convert the canvas to a raw RGB buffer
    # Create a bytes buffer to save the plot
    buf = io_builtin.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    # Open the PNG image from the buffer and convert it to a NumPy array
    image = np.array(Image.open(buf))
    plt.close(fig)
    return image


if __name__ == "__main__":
    screen = "timecourse"
    file = "/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/configs/params.yaml"

    stain_metadata = pd.read_csv(
        "/pstore/data/ihb-g-deco/USERS/schulzp9/tumoroid/metafiles/timecourse_stainings_metadata.csv"
    )
    stain_metadata["staining_set"] = stain_metadata["staining_set"].astype(str)
    stains = stain_metadata.stain.unique()
    stain_dict = {
        set_id: dict(group[["channel", "stain"]].values)
        for set_id, group in stain_metadata.groupby("staining_set")
    }

    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    df_plate_layout = pd.read_csv(Path(params["dir_screen"], "timecourse_layout.csv"))
    df_plate_layout = df_plate_layout.melt(
        id_vars=["row"], var_name="col", value_name="staining_set"
    )
    df_plate_layout["column"] = df_plate_layout["col"].str.zfill(2)
    df_plate_layout["well_id"] = df_plate_layout["row"] + df_plate_layout["column"]

    dir_screen = params["dir_screen"]
    plates = params["plates"]
    df_plates = pd.concat(
        [
            pd.read_csv(
                f"{dir_screen}/{plate}/plate_information.csv", dtype={plate: str}
            ).assign(plate=plate)
            for plate in plates
        ]
    )

    df_plate_layouts = df_plate_layout.merge(df_plates, how="cross")
    df_plate_layouts["plate_id"] = df_plate_layouts["plate"]
    df_stain_layout = df_plate_layouts[["well_id", "plate_id", "staining_set"]]
    df_stain_layout = df_stain_layout.applymap(str)

    # get nuclei channel columns
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
        df_plates["label"].astype(str)
        + "_"
        + df_plates["well"]
        + "_"
        + df_plates["plate"]
    )

    mdata = mu.read_h5mu(
        "/pmount/projects/site/pred/ihb-tumoroid/data/processed/timecourse/anndata/mdata_registered_imputed_mlp_normalized.h5mu"
    )

    dir_plots = "/pstore/home/harmelc/tumoroid_screen/whole_mount_tumoroid/analysis/timecourse/plots"
    wells = {"001": "H05", "002": "H05", "003": "H05", "004": "G05", "005": "G05"}
    features = ["SDC1", "YAP", "Phalloidin"]
    mods = {
        "neighbors": "phenocoder_msg_neighbors_imputed",
        "nuclei": "phenocoder_msg_nuclei_imputed",
    }
    for mod in mods.keys():
        adata = mdata[mods[mod]].copy()
        obs_df = data_setup(adata, df_stain_layout, df_plates, mod, stain_dict)

        for feature in features:
            imgs = []
            imgs_pred = []
            for plate_id, well_id in wells.items():
                id = f"{well_id}_{plate_id}"
                img = plot_organoid_feature(
                    id,
                    mdata[mods[mod]],
                    feature,
                    project_3d=True,
                    cut_open=True,
                    edgecolors="grey",
                    bg_color="white",
                    obs_df=obs_df,
                    plot_obs_df=True,
                )
                imgs.append(img)
                img_pred = plot_organoid_feature(
                    id,
                    mdata[mods[mod]],
                    feature,
                    project_3d=True,
                    cut_open=True,
                    edgecolors="grey",
                    bg_color="white",
                    obs_df=obs_df,
                    plot_obs_df=False,
                )
                imgs_pred.append(img_pred)

            montage_measured = montage(
                np.asarray(imgs), channel_axis=-1, grid_shape=(1, len(wells))
            )
            montage_pred = montage(
                np.asarray(imgs_pred), channel_axis=-1, grid_shape=(1, len(wells))
            )

            io.imsave(
                f"{dir_plots}/predictions_spatial_{mod}_{feature}.png", montage_pred
            )
            io.imsave(
                f"{dir_plots}/measured_spatial_{mod}_{feature}.png", montage_measured
            )
