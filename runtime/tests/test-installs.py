import logging
import importlib

logging.getLogger('').setLevel(logging.INFO)

logging.info("Testing if Python packages can be loaded correctly.")

packages = [
    "cv2",  # opencv
    "dotenv",
    "fastai",
    "lightgbm",
    "mahotas",
    "numpy",
    "pandas",
    "PIL",  # pillow
    "pyvips",
    "skimage",  # scikit-image
    "sklearn",  # scikit-learn
    "scipy",
    "tensorflow",
    "torch",  # pytorch
    "torchvision",
    "xgboost",
    # ADD ADDITIONAL REQUIREMENTS BELOW HERE #
    "openslide",
    "imagecodecs"
    ##########################################
]

for package in packages:
    logging.info("Testing if {} can be loaded...".format(package))
    importlib.import_module(package)

logging.info("All required packages successfully loaded.")
