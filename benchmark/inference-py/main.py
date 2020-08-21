from datetime import datetime
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
import torchvision

# We get to see the log output for our execution, so log away!
logging.basicConfig(level=logging.INFO)

ROOT_DIRECTORY = Path(__file__).parent.expanduser().resolve()
MODEL_PATH = ROOT_DIRECTORY / "assets" / "my-awesome-model.pt"

# The images will live in a folder called '/inference/data/test_images' in the container
DATA_DIRECTORY = Path("/inference/data")
IMAGE_DIRECTORY = DATA_DIRECTORY / "test_images"


def vips2numpy(fetch, image_format, width, height, bands):
    format_to_dtype = {
        "uchar": np.uint8,
        "char": np.int8,
        "ushort": np.uint16,
        "short": np.int16,
        "uint": np.uint32,
        "int": np.int32,
        "float": np.float32,
        "double": np.float64,
        "complex": np.complex64,
        "dpcomplex": np.complex128,
    }
    return np.ndarray(
        buffer=fetch, dtype=format_to_dtype[image_format], shape=[height, width, bands],
    )


class WholeSlideImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path: Path,
        image_directory: Path = IMAGE_DIRECTORY,
        tile_width: int = 512,
        tile_height: int = 512,
        transform=None,
    ):
        metadata = pd.read_csv(metadata_path, index_col="filename")

        indices = []
        regions = []
        for image_index, entry in enumerate(metadata.itertuples()):
            image = pyvips.Image.new_from_file(
                str(IMAGE_DIRECTORY / Path(entry.Index).with_suffix(".tif"))
            )
            regions.append(
                {
                    "region": pyvips.Region.new(image),
                    "format": image.format,
                    "bands": image.bands,
                }
            )
            for row, column in itertools.product(
                range(entry.width // (tile_width - 1)),
                range(entry.height // (tile_height - 1)),
            ):
                indices.append(
                    {
                        "filename": Path(entry.Index).with_suffix(".tif"),
                        "image_index": image_index,
                        "row": row,
                        "column": column,
                    }
                )

        logging.info(
            "Dataset of %d images (%d tiles) images from %s",
            len(metadata),
            len(indices),
            image_directory,
        )
        self.indices = pd.DataFrame(indices)
        self.regions = regions
        self.image_directory = image_directory
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.transform = transform
        self._loaded_image = None
        self._loaded_image_path = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices.iloc[index]
        region = self.regions[index.image_index]["region"]
        try:
            tile = region.fetch(
                index.column * self.tile_width,
                index.row * self.tile_height,
                self.tile_width,
                self.tile_height,
            )
        # until we fix the width and height in the metadata
        except pyvips.error.Error:
            tile = region.fetch(0, 0, self.tile_width, self.tile_height)

        tile = vips2numpy(
            tile,
            self.regions[index.image_index]["format"],
            self.tile_width,
            self.tile_height,
            self.regions[index.image_index]["bands"],
        )

        if self.transform is not None:
            tile = self.transform(tile)

        return tile, index.filename.name


def perform_inference(batch_size: int = 16):
    """This is the main function executed at runtime in the cloud environment.
    """
    logging.info("Loading model.")
    model = torch.load(str(MODEL_PATH))
    if torch.cuda.is_available():
        model = model.to("cuda")

    logging.info("Loading and processing metadata.")

    transform = torchvision.transforms.ToTensor()
    dataset = WholeSlideImageDataset(
        DATA_DIRECTORY / "test_metadata.csv", transform=transform
    )

    logging.info("Starting inference.")
    # Preallocate prediction output
    submission_format = pd.read_csv(
        DATA_DIRECTORY / "submission_format.csv", index_col="filename"
    )

    data_generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    # Perform (and time) inference
    inference_start = datetime.now()
    logging.info(
        "Starting inference %s (%d batches)",
        inference_start,
        len(dataset) // batch_size,
    )
    predictions = []
    for batch_index, (batch, slide) in enumerate(data_generator):
        logging.info("Batch %d %s", batch_index, datetime.now())
        if torch.cuda.is_available():
            batch = batch.to("cuda")
        with torch.no_grad():
            preds = model.forward(batch)
        for label in preds.argmax(1):
            predictions.append({"label": int(label), "slide": slide})

    inference_end = datetime.now()
    logging.info(
        "Inference complete at %s (duration %s)",
        inference_end,
        inference_end - inference_start,
    )

    # Check our predictions are in the same order as the submission format
    predictions = pd.DataFrame(predictions)
    submission = predictions.groupby("slide").label.max()
    logging.info("Creating submission.")
    submission = submission.loc[submission_format.index]
    assert (submission.index == submission_format.index).all()

    # # We want to ensure all of our data are floats, not integers
    submission = submission.astype(np.float)

    # Save out submission to root of directory
    submission.to_csv("submission.csv", index=True)
    logging.info("Submission saved.")


if __name__ == "__main__":
    perform_inference(batch_size=512)
