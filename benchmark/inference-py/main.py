from datetime import datetime
import itertools
import logging
from pathlib import Path

import pandas as pd
import PIL
import pyvips
import torch
import torchvision

# We get to see the log output for our execution, so log away!
logging.basicConfig(level=logging.INFO)

ROOT_DIRECTORY = Path(__file__).parent.expanduser().resolve()
MODEL_PATH = ROOT_DIRECTORY / "assets" / "my-awesome-model.pt"

# The images will live in a folder called '/inference/data/test_images' in the container
DATA_DIRECTORY = Path("/inference/data")
IMAGE_DIRECTORY = DATA_DIRECTORY / "test_images" / "annotated_sample_tifs"
TILE_DIRECTORY = ROOT_DIRECTORY / "tiles"


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
        for entry in metadata.itertuples():

            for row, column in itertools.product(
                range(entry.width // tile_width), range(entry.height // tile_height)
            ):
                indices.append(
                    {
                        "filename": Path(entry.Index).with_suffix(".tif"),
                        "row": row,
                        "column": column,
                    }
                )

        logging.info(
            "Dataset of %s images from %s", len(indices), image_directory,
        )
        self.indices = pd.DataFrame(indices)
        self.image_directory = image_directory
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices.iloc[index]
        logging.info("Reading %s %d %d", index.filename, index.column, index.row)
        image = pyvips.Image.new_from_file(
            str(self.image_directory / index.filename)
        ).crop(
            index.column * self.tile_width,
            index.row * self.tile_height,
            self.tile_width,
            self.tile_height,
        )
        image = PIL.Image.frombuffer(
            "RGB", (self.tile_width, self.tile_height), image.write_to_memory()
        )
        if self.transform is not None:
            image = self.transform(image)

        return image


def perform_inference():
    """This is the main function executed at runtime in the cloud environment.
    """
    logging.info("Loading model.")
    model = torch.load(str(MODEL_PATH))

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

    data_generator = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Perform (and time) inference
    inference_start = datetime.now()
    logging.info("Starting inference %s", inference_start)
    predictions = []
    for batch in data_generator:
        preds = model.forward(batch)
        for label in preds.argmax(1):
            predictions.append({"label": int(label)})

    inference_end = datetime.now()
    logging.info(
        "Inference complete at %s (duration %s)",
        inference_end,
        inference_end - inference_start,
    )

    # # Check our predictions are in the same order as the submission format
    # predictions = pd.DataFrame(predictions)
    # submission = predictions.groupby("slide").label.max()
    # logging.info("Creating submission.")
    # submission = submission.loc[submission_format.index]
    # assert (submission.index == submission_format.index).all()

    # # We want to ensure all of our data are floats, not integers
    # submission = submission.astype(np.float)

    # Save out submission to root of directory
    # submission.to_csv("submission.csv", index=True)
    submission_format.to_csv("submission.csv", index=True)
    logging.info("Submission saved.")


if __name__ == "__main__":
    perform_inference()
