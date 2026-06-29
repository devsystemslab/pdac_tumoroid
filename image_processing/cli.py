from pathlib import Path

import pandas as pd
import yaml

from image_processing import (
    background,
    features,
    flatfield,
    mip,
    montage,
    neighborhoods,
    preprocessing,
    registration,
    segmentation,
    stitching3d,
    utils,
)


def default_config() -> dict:
    def_config = {
        "screen": "screen_name",
        "dir_output": "path/to/output",
        "metadata": "metadata.csv",
        "plate_layout": "plate_layout.csv",
        "clean_up": False,
        "overwrite_existing": False,
        "overlap": 0.1,
        "wells": None,
        "preprocessing": {
            "run": True,
            "illuminaton_correction": True,
            "remove_first_tile": True,
        },
        "background_correction": {
            "run": True,
            "mode": "basicpy",  # 'basic' or 'smo'
            "multiplicative": {
                "plate_id_x": ["channel0d"],
                "plate_id_y": ["channel0d"],
            },
            "median_filter": {"plate_id_x": ["channel0d"], "plate_id_y": ["channel0d"]},
        },
        "mip": {"run": True, "input": ["TIF_OVR", "TIF_OVR_BG"]},
        "montage": {"run": True, "input": ["TIF_MIP_OVR", "TIF_MIP_OVR_BG"]},
        "segmentation": {"run": True, "restore_dapi": False, "input": ["TIF_OVR_BG"]},
        "feature_extraction": {
            "run": True,
            "input": ["TIF_OVR", "TIF_OVR_BG"],
            "segmentation_input": "SEG_TIF_OVR_BG",
        },
        "registration": {
            "run": True,
            "input": ["TIF_OVR_BG"],
            "source": "01",
            "target": "03",
            "z_step": 10,
            "pixel_size": 0.322,
        },
    }
    return def_config


def write_default_config(config_file: str) -> None:
    with open(config_file, "w") as f:
        yaml.dump(default_config(), f, sort_keys=False)


def load_config(config_file: str) -> dict:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def run_pipeline(plate: str, config_file: str) -> None:
    """
    Run the complete image processing pipeline for a single plate
    :param plate:
    :param config_file:
    :return:
    """
    # load config
    if config_file is None:
        config = default_config()
    else:
        config = load_config(config_file)

    # load metadata file and select plate_id
    df_metadata = pd.read_csv(config["metadata"])
    df_metadata = df_metadata[
        (df_metadata["plate"] == plate) & (df_metadata["screen"] == config["screen"])
    ]
    assert len(df_metadata) >= 1, (
        f"Plate {plate} not found in metadata or for {config['screen']}!"
    )
    df_metadata = df_metadata.dropna(subset=["dir_raw_images"])

    # process plates
    plate_ids = sorted(df_metadata["plate_id"].unique().tolist())
    df_plates = []
    for plate_id in plate_ids:
        # check if plate is already processed
        if Path(
            config["dir_output"], config["screen"], plate, "plate_information.csv"
        ).is_file():
            df_plates_processed = pd.read_csv(
                Path(
                    config["dir_output"],
                    config["screen"],
                    plate,
                    "plate_information.csv",
                ),
                dtype={"plate": str},
            )
            df_processed = df_plates_processed[
                (df_plates_processed["plate_id"] == plate_id)
            ]
            if len(df_processed) > 0:
                df_plates.append(df_processed)
                if config["overwrite_existing"]:
                    print(
                        f"Plate {plate_id} already processed. Overwriting existing data..."
                    )
                else:
                    print(f"Plate {plate_id} already processed. Skipping...")
                    continue
            else:
                print(f"Plate {plate_id} not processed yet. Starting processing...")

        # select metadata for plate_id
        print(f"Processing plate {plate_id}...")
        df_metadata_selected = df_metadata[df_metadata["plate_id"] == plate_id]
        assert len(df_metadata_selected) == 1, (
            f"Plate ID {plate_id} not found in metadata or not unique for {config['screen']}!"
        )

        # set up output directory
        dir_plate = Path(config["dir_output"], config["screen"], plate, plate_id)
        dir_plate.mkdir(parents=True, exist_ok=True)

        # load plate_information
        df_plate = utils.load_plate_information(
            screen=config["screen"],
            dir_processed=config["dir_output"],
            dir_images=df_metadata_selected["dir_images"].iloc[0],
            tile_overlap=config["overlap"],
            plate=plate,
            plate_id=plate_id,
            dir_raw_imgs=df_metadata_selected["dir_raw_images"].iloc[0],
            cycle=df_metadata_selected["cycle"].iloc[0],
            ch01=df_metadata_selected["ch01"].iloc[0],
            ch02=df_metadata_selected["ch02"].iloc[0],
            ch03=df_metadata_selected["ch03"].iloc[0],
            ch04=df_metadata_selected["ch04"].iloc[0],
        )

        # add start time
        start = pd.Timestamp.now()
        df_plate["start"] = start.strftime("%Y-%m-%d %H:%M")

        # run preprocessing
        if config["preprocessing"]["run"]:
            if config["wells"] is None:
                df_images = utils.get_metadata(df_plate["dir_raw"].iloc[0])
                config["wells"] = [df_images["well_id"].unique().tolist()]
            for i, wells in enumerate(config["wells"]):
                print(f"Processing well set {i + 1}: {len(wells)} wells...")
                # run flatfield calculation
                if config["preprocessing"]["illuminaton_correction"]:
                    print("Running flatfield estimation...")
                    flatfield.run_basic(
                        dir_images=df_plate["dir_raw"].iloc[0],
                        chunk_size=config["preprocessing"]["chunk_size"],
                        dir_plate=dir_plate,
                        remove_first_tile=config["preprocessing"]["remove_first_tile"],
                        wells=wells,
                        max_iter=config["preprocessing"]["max_iter"],
                    )

                print("Running preprocessing...")
                preprocessing.process_plate(
                    df_plate=df_plate,
                    dir_plate=dir_plate,
                    remove_first_tile=config["preprocessing"]["remove_first_tile"],
                    wells=wells,
                )

        # run background estimation
        if config["background_correction"]["run"]:
            print("Running background correction...")
            background.process_plate(
                dir_plate,
                input_type="TIF_OVR",
                config=config["background_correction"],
                df_metadata_plate=df_metadata_selected,
            )

        # run mip
        if config["mip"]["run"]:
            print("Running MIP...")
            for input_type in config["mip"]["input"]:
                mip.process_plate(dir_plate, input_type=input_type)

        # run montages
        if config["montage"]["run"]:
            print("Running montages...")
            for input_type in config["montage"]["input"]:
                montage.generate_montages(dir_plate, config["plate_layout"], input_type)

        # run nuclei segmentation
        if config["segmentation"]["run"]:
            for input_type in config["segmentation"]["input"]:
                print(f"Running segmentation for {input_type}...")
                segmentation.process_plate(
                    dir_plate, input_type, config["segmentation"]["restore_dapi"]
                )
                if input_type.startswith("TIF_OVR"):
                    print("Running 3D stitching...")
                    stitching3d.process_plate(dir_plate, f"SEG_{input_type}")

        # run feature extraction
        if config["feature_extraction"]["run"]:
            for input_type in config["feature_extraction"]["input"]:
                print(f"Running feature extraction for {input_type}...")
                features.process_plate(
                    dir_plate,
                    input_type,
                    config["feature_extraction"]["segmentation_input"],
                )
                neighborhoods.process_plate(
                    dir_plate,
                    input_type,
                    config["feature_extraction"]["segmentation_input"],
                )

        # add end time
        end = pd.Timestamp.now()
        df_plate["end"] = end.strftime("%Y-%m-%d %H:%M")
        # add duration
        df_plate["duration"] = str(end - start)
        df_plates.append(df_plate)

    # save plate information
    if len(df_plates) > 0:
        print("Saving plate information...")
        df_plates = pd.concat(df_plates, axis=0)
        df_plates.to_csv(
            Path(
                config["dir_output"], config["screen"], plate, "plate_information.csv"
            ),
            index=False,
        )

    # run registration
    if config["registration"]["run"]:
        for input_type in config["registration"]["input"]:
            print(f"Running registration for {input_type}...")
            registration.process_plate(
                Path(config["dir_output"], config["screen"], plate),
                source_cycle=f"{plate}-{config['registration']['source']}",
                target_cycle=f"{plate}-{config['registration']['target']}",
                input_type=input_type,
                z_step=config["registration"]["z_step"],
                pixel_size=config["registration"]["pixel_size"],
            )
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tumoroid image processing pipeline")
    parser.add_argument("plate", help="ID of the plate to process")
    parser.add_argument("--config", default=None, help="Path to the configuration file")
    args = parser.parse_args()
    run_pipeline(args.plate, args.config)
