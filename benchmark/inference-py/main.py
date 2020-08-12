from datetime import datetime
import logging
from pathlib import Path
import random

import pandas as pd
import pyvips
import numpy as np
import torch
import torchvision

# We get to see the log output for our execution, so log away!
logging.basicConfig(level=logging.INFO)

ROOT_DIRECTORY = Path(__file__).parent.expanduser().resolve()
# ROOT_DIRECTORY = Path(".").expanduser().resolve() / "benchmark" / "inference-py"
ASSET_DIRECTORY = ROOT_DIRECTORY / "assets"
MODEL_PATH = ASSET_DIRECTORY / "my-awesome-model.pt"

# The images will live in a folder called 'data' in the container
DATA_DIRECTORY = Path("/inference/data")
# DATA_DIRECTORY = (
#     Path("~/projects/sfp-cervical-biopsy-runtime/inference-data").expanduser().resolve()
# )
TILE_DIRECTORY = ROOT_DIRECTORY / "tiles"


def generate_image_tiles(
    input_directory: Path = DATA_DIRECTORY,
    output_directory: Path = TILE_DIRECTORY,
    tile_width: int = 512,
    tile_height: int = 512,
    tiles_per_level: int = 10,
):
    slide_indices = {}
    for slide_index, path in enumerate(sorted(input_directory.glob("*.tif"))):
        slide_indices[slide_index] = path.stem
        (output_directory / path.stem).mkdir(exist_ok=True, parents=True)
        for level in range(0, 3):
            image = pyvips.Image.new_from_file(str(path), page=level)
            for tile_index in range(tiles_per_level):
                x = random.randrange(0, image.width - tile_width)
                y = random.randrange(0, image.height - tile_height)
                output_path = (
                    output_directory / path.stem / f"level{level}_tile{tile_index}.tif"
                )
                image.crop(x, y, tile_width, tile_height).tiffsave(
                    str(output_path), tile=False
                )
    return slide_indices


def perform_inference():
    """This is the main function executed at runtime in the cloud environment.
    """
    logging.info("Loading model.")
    model = torch.load(MODEL_PATH)

    logging.info("Loading and processing metadata.")

    # Our preprocessing selects the first image for each sequence
    test_metadata = pd.read_csv(
        DATA_DIRECTORY / "test_metadata.csv", index_col="seq_id"
    )
    test_metadata = (
        test_metadata.sort_values("file_name").groupby("seq_id").first().reset_index()
    )

    # Prepend the path to our filename since our data lives in a separate folder
    test_metadata["full_path"] = test_metadata.file_name.map(
        lambda x: str(DATA_DIRECTORY / x)
    )

    logging.info("Starting inference.")

    # Preallocate prediction output
    submission_format = pd.read_csv(
        DATA_DIRECTORY / "submission_format.csv", index_col="slide"
    )

    # Tile input images
    slide_indices = generate_image_tiles()

    transform = torchvision.transforms.ToTensor()
    image_folder = torchvision.datasets.ImageFolder(TILE_DIRECTORY, transform=transform)
    data_generator = torch.utils.data.DataLoader(
        image_folder, batch_size=4, shuffle=False
    )

    # Perform (and time) inference
    inference_start = datetime.now()
    predictions = []
    for batch, slides in data_generator:
        preds = model.forward(batch)
        for label, slide in zip(preds.argmax(1), slides):
            predictions.append(
                {"label": int(label), "slide": slide_indices[int(slide)]}
            )

    logging.info(f"Inference complete. Took {datetime.now() - inference_start}.")

    # Check our predictions are in the same order as the submission format
    predictions = pd.DataFrame(predictions)
    submission = predictions.groupby("slide").label.max()
    logging.info("Creating submission.")
    submission = submission.loc[submission_format.index]
    assert (submission.index == submission_format.index).all()

    # We want to ensure all of our data are floats, not integers
    submission = submission.astype(np.float)

    # Save out submission to root of directory
    submission.to_csv("submission.csv", index=True)
    logging.info("Submission saved.")


if __name__ == "__main__":
    perform_inference()
