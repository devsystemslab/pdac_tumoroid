import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
import yaml

# set matplotlib fontconfig 42
from matplotlib import rc

from whole_mount_tumoroid.phenocoder.spatial import get_chull, get_moran
from whole_mount_tumoroid.phenocoder.utils import load_plate

rc("pdf", fonttype=42)

df_stains = pd.read_csv("metafiles/timecourse_stainings_metadata.csv")
df_plate_layout = pd.read_csv("metafiles/timecourse_layout.csv")

df_plate_layout = df_plate_layout.melt(
    id_vars=["row"], var_name="col", value_name="staining_set"
)
df_plate_layout["col"] = df_plate_layout["col"].str.zfill(2)
df_plate_layout["well"] = df_plate_layout["row"] + df_plate_layout["col"]
df_plate_layout = df_plate_layout.merge(df_stains, on=["staining_set"])
df_plate_layout["channel"] = df_plate_layout["channel"].astype(str).str.zfill(2)

dir_screen = "data/processed/timecourse"
plates = ["001", "002", "003", "004", "005"]
df_plates = pd.concat(
    [
        pd.read_csv(
            f"{dir_screen}/{plate}/plate_information.csv", dtype={plate: str}
        ).assign(plate=plate)
        for plate in plates
    ]
)
input_type = "TIF_OVR_BG"
df = pd.concat(
    [
        load_plate(
            plate,
            input_type,
            dir_screen,
            registered=False,
            plate_id=plate_id,
            z_step=10,
        )
        for plate, plate_id in zip(df_plates["plate"], df_plates["plate_id"])
    ]
)
# reset index
df = df.reset_index(drop=False)
# label + well + plate
df["label_id"] = (
    df["label"].astype(str) + "_" + df["well"] + "_" + df["plate"].astype(str)
)
# merge with plate layout relation many-to-many
df = df.merge(
    df_plate_layout[["staining_set", "well"]].drop_duplicates(), on="well", how="left"
)

# load qc yaml file
with open("sc/qc.yaml") as file:
    qc = yaml.load(file, Loader=yaml.FullLoader)
df_qc = pd.concat(
    [
        pd.DataFrame({"plate_id": key, "well": value})
        for key, value in qc["timecourse"].items()
    ]
)
# remove -01 from plate_id
df_qc["plate"] = df_qc["plate_id"].str.split("-").str[0]
# filter out df with df_qc anti-join on plate_id, well
df = df.merge(df_qc, on=["plate", "well"], how="outer")
# keep rows where plate_id is NaN and drop column
df = df[df["plate_id"].isna()].drop(columns=["plate_id"])
df.set_index("label_id", inplace=True)
adata = ad.AnnData(df.filter(regex="_neighbors"))
adata.obs = df.drop(columns=df.filter(regex="_neighbors").columns)
df_iter = df.groupby(["well", "plate"]).size().reset_index()
volumes = [
    get_chull_volume(well, plate, adata)
    for well, plate in zip(df_iter["well"], df_iter["plate"])
]
df_volumes = pd.DataFrame(
    {"volume": volumes, "well": df_iter["well"], "plate": df_iter["plate"]}
)
# dict timepoints vs plate
timepoints = {
    "001": "day3",
    "002": "day5",
    "003": "day7",
    "004": "day10",
    "005": "day14",
}
df_volumes["timepoint"] = df_volumes["plate"].map(timepoints)
# convert volume to um^3 -> pixel size is 0.322 um pro pixel
df_volumes["volume"] = df_volumes["volume"] * 0.322**3
# convert to cubic mm
df_volumes["volume"] = df_volumes["volume"] / 1000**3
df_volumes["timepoint_int"] = df_volumes["timepoint"].str.extract("(\d+)").astype(int)
# save volumes
df_volumes.to_csv(f"{dir_screen}/df_volumes.csv", index=False)

# get cell counts
df_cell_counts = df.groupby(["well", "plate"]).size().reset_index()
df_cell_counts.columns = ["well", "plate", "cell_count"]
df_cell_counts["timepoint"] = df_cell_counts["plate"].map(timepoints)
df_cell_counts["timepoint_int"] = (
    df_cell_counts["timepoint"].str.extract("(\d+)").astype(int)
)
# save cell counts
df_cell_counts.to_csv(f"{dir_screen}/df_cell_counts.csv", index=False)


# merge with volumes and cell counts
df_volumes = df_volumes.merge(
    df_cell_counts, on=["well", "plate", "timepoint", "timepoint_int"]
)
df_volumes["density"] = df_volumes["cell_count"] / df_volumes["volume"]
# set seaborn style white
sns.set_style("ticks")
# same but with jitter plot
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
sns.swarmplot(x="timepoint", y="volume", data=df_volumes, ax=ax[0, 0])
# add y axis label
ax[0, 0].set_ylabel("Volume (mm^3)")
# add regression bspline
sns.regplot(
    x="timepoint_int",
    y="volume",
    data=df_volumes,
    lowess=True,
    line_kws={"color": "C1"},
    ax=ax[1, 0],
    x_jitter=0.5,
)
ax[1, 0].set_ylabel("Volume (mm^3)")
ax[1, 0].set_xlabel("Timepoint [days]")

sns.swarmplot(x="timepoint", y="cell_count", data=df_cell_counts, ax=ax[0, 1])
ax[0, 1].set_ylabel("Cell count")
# add regression bspline
sns.regplot(
    x="timepoint_int",
    y="cell_count",
    data=df_cell_counts,
    lowess=True,
    line_kws={"color": "C1"},
    ax=ax[1, 1],
    x_jitter=0.5,
)
ax[1, 1].set_ylabel("Cell count")
ax[1, 1].set_xlabel("Timepoint [days]")


sns.swarmplot(x="timepoint", y="density", data=df_volumes, ax=ax[0, 2])
ax[0, 2].set_ylabel("Density (cells/mm^3)")
# set y lims to 0 and 99th percentile
# ax[0,2].set_ylim(0, df_volumes['density'].quantile(0.99))


# add regression bspline
sns.regplot(
    x="timepoint_int",
    y="density",
    data=df_volumes,
    lowess=True,
    line_kws={"color": "C1"},
    ax=ax[1, 2],
    x_jitter=0.5,
)
ax[1, 2].set_ylabel("Density (cells/mm^3)")
# ax[1,2].set_ylim(0, df_volumes['density'].quantile(0.99))
# specify positions of ticks on x-axis and y-axis
# set x axis labels
ax[1, 2].set_xlabel("Timepoint [days]")
plt.tight_layout()
# save to pdf
plt.savefig(f"{dir_screen}/timecourse_plots.pdf")
plt.show()


# now just with box plots for volume and cell count and remove day3
df_volumes = df_volumes[df_volumes["timepoint"] != "day3"]
df_cell_counts = df_cell_counts[df_cell_counts["timepoint"] != "day3"]
# remove cell counts smaller 2000
df_cell_counts = df_cell_counts[df_cell_counts["cell_count"] > 2000]
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.boxplot(x="timepoint", y="volume", data=df_volumes, ax=ax[0])
ax[0].set_ylabel("Volume (mm^3)")
sns.boxplot(x="timepoint", y="cell_count", data=df_cell_counts, ax=ax[1])
ax[1].set_ylabel("Cell count")
# set width and hight of pdf
plt.rcParams["lines.linewidth"] = 0.5
# fontsize to 6
plt.rcParams.update({"font.size": 6})
# markersize smaller
plt.rcParams.update({"lines.markersize": 2})
plt.gcf().set_size_inches(5, 1.25)
plt.savefig(f"{dir_screen}/timecourse_boxplots.pdf", dpi=72)
plt.show()

results = []
for staining_set in df["staining_set"].unique():
    print(f"Processing staining set: {staining_set}")
    df_tmp = df[df["staining_set"] == staining_set].copy()
    df_tmp = df_tmp.drop(columns=["staining_set"])
    adata_tmp = ad.AnnData(df_tmp.filter(regex="_neighbors"))
    adata_tmp.obs = df_tmp.drop(columns=df_tmp.filter(regex="_neighbors").columns)
    adata_tmp = sc.pp.regress_out(adata_tmp, ["z"], copy=True)
    sc.pp.scale(adata_tmp)
    # save adata_tmp
    adata_tmp.write(f"{dir_screen}/adata_set_{staining_set}.h5ad")
    df_iter = df_tmp.groupby(["well", "plate"]).size().reset_index()
    df_moran = []
    for well, plate in zip(df_iter["well"], df_iter["plate"]):
        print(f"Processing well: {well} - {plate}")
        adata_tmp_well = adata_tmp[adata_tmp.obs["well"] == well]
        adata_tmp_well = adata_tmp_well[adata_tmp_well.obs["plate"] == plate]
        # generate spatial obsm
        adata_tmp_well.obsm["spatial3d"] = adata_tmp_well.obs[
            ["centroid-0", "centroid-1", "z"]
        ].values
        sq.gr.spatial_neighbors(
            adata_tmp_well, radius=100, coord_type="generic", spatial_key="spatial3d"
        )
        df_moran.append(get_moran(adata_tmp_well, well, plate))
    df_moran = pd.concat(df_moran)
    # add staining set
    df_moran["staining_set"] = staining_set
    # pivot longer
    df_moran.reset_index(inplace=True)
    # rename column 'index' to label_id
    df_moran.rename(columns={"index": "label_id"}, inplace=True)
    df_moran = df_moran.melt(
        id_vars=["staining_set", "label_id"], var_name="metric", value_name="value"
    )
    # rename metric to channel
    df_moran["channel"] = df_moran["metric"].str.split("_").str[1]
    # merge with df_plate_layout
    df_moran = df_moran.merge(
        df_plate_layout[["staining_set", "channel", "stain"]],
        on=["staining_set", "channel"],
        how="left",
    )
    # filter out Nan values
    df_moran = df_moran[~df_moran["stain"].isna()]
    results.append(df_moran)

df_moran = pd.concat(results)
# remove white spaces from stain column
df_moran["stain"] = df_moran["stain"].str.strip()
# split label id into label, well, plate
df_moran[["well", "plate"]] = df_moran["label_id"].str.split("_", expand=True)
df_moran["timepoint"] = df_moran["plate"].map(timepoints)
# save moran's I
df_moran.to_csv(f"{dir_screen}/df_morans.csv", index=False)
# plot moran's I as heatmap timepoint vs stain in facets doing beeswarm plot
# set seaborn style white
sns.set_style("ticks")
df_plot = df_moran.groupby(["timepoint", "stain"]).mean(["value"]).reset_index()
# drop 'staining_set' columnvalues
df_plot = df_plot.drop(columns="staining_set")
df_plot["timepoint"] = pd.Categorical(
    df_plot["timepoint"],
    categories=["day3", "day5", "day7", "day10", "day14"],
    ordered=True,
)
# pivot to rows = stains and cols = timepoints
df_plot = df_plot.pivot(index="timepoint", columns="stain", values="value")
df_plot = df_plot.sort_values("timepoint")
# heatmap
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(df_plot, cmap="coolwarm", ax=ax)
plt.tight_layout()
plt.show()
# do cluster map, clustering columns, keep rows in order

sns.clustermap(df_plot, cmap="coolwarm", row_cluster=False, figsize=(24, 7))
# add text
plt.text(
    -0.1,
    -1,
    "Moran's I at 100 pixel kernel size\n for spatial neighborhoods",
    fontsize=16,
    rotation=0,
)
# save fig
plt.savefig(f"{dir_screen}/moran_heatmap.pdf")
plt.show()
