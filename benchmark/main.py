from pathlib import Path
import random

import pandas as pd


DATA_ROOT = Path(__file__).parent / "data"


def perform_inference(input_metadata, submission_format):
    for filename, metadata_row in input_metadata.iterrows():
        # make sure all the image exists
        assert (DATA_ROOT / filename).exists()

        # make a random prediction
        pred = random.choice([0, 1, 2, 3])

        # one-hot encode the prediction
        submission_format.loc[filename, pred] = 1

    # save as "submission.csv" in the root folder, where it is expected
    submission_format.to_csv("submission.csv")

if __name__ == "__main__":
    # load metadata
    input_metadata = pd.read_csv(
        DATA_ROOT / 'test_metadata.csv',
        index_col=0
    )

    # load sumission format
    submission_format = pd.read_csv(
        DATA_ROOT / 'submission_format.csv',
        index_col=0
    )
    
    perform_inference(input_metadata, submission_format)
