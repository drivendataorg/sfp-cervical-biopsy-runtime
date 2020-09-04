# TissueNet: Detect Lesions in Cervical Biopsies

Welcome to the runtime repository for the [SFP Cervical Biopsy challenge](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/). This repo contains the definition of the environment where your code submissions will run. It specifies both the operating system and the Python packages that will be available to your solution.

This repository has two primary uses for competitors:

 - (1) It lets you test your `submission.zip` file with a locally running version of the container so you don't have to wait for it to process on the competition site to find programming errors.
 - (2) It lets you test adding additional Python packages to the official runtime environment. You can then submit a PR to request those packages be included in the container image.

### [Getting started](#0-getting-started)
 - [Prerequisites](#prerequisites)
 - [Quickstart](#quickstart)
### [Testing your submission.zip](#1-testing-your-submissionzip)
 - [Implement your solution](#implement-your-solution)
 - [How your submission will run](#how-your-submission-will-run)
 - [Test running your submission locally](#test-running-your-submission-locally)
   - [Making a submission](#making-a-submission)
   - [Reviewing the logs](#reviewing-the-logs)
### [Updating the runtime packages](#2-updating-the-runtime-packages)
 - [Adding new Python packages](#adding-dependencies-to-the-runtime)
 - [Testing new dependencies](#testing-new-dependencies)
 - [Submitting a PR](#opening-a-pull-request)


## (0) Getting started

### Prerequisites

Make sure you have the prerequisites installed.

 - A clone or fork of this repository
 - docker
 - GNU make (optional, but useful for using the commands in the Makefile)
 - At least ~10GB of free space for both the training images and the Docker container images

Additional requirements to run with GPU:
 - NVIDIA drivers and container runtime: [Follow these instructions](https://docs.docker.com/config/containers/resource_constraints/#gpu)

### Quickstart

To test out the full execution pipeline, run the following commands in order in the terminal. These will get the Docker images, download a few training images with which to test the execution, zip up an example submission script, and submit that submission.zip to your locally running version of the container.

```
make pull
make sample-images
make pack-benchmark
make test-submission
```

You should see output like this in the end (and find the same logs in the folder `submission/log.txt`):

```
TODO: log output
```

Running `make` at the terminal will tell you all the commands available in the repository:

```
Settings based on your machine:
CPU_OR_GPU=gpu 			# Whether or not to try to build, download, and run GPU versions
SUBMISSION_IMAGE=8edab2591180 	# ID of the image that will be used when running test-submission

Available competition images:
drivendata/sfp-competition:gpu-latest(297317990f76}), drivendata/sfp-competition:gpu-local(8edab2591180}),

Available commands:

build               Builds the container locally, tagging it with cpu-local or gpu-local
debug-container     Start your locally built container and open a bash shell within the running
                    container; same as submission setup except has network access
pack-benchmark      Creates a submission/submission.zip file from whatever is in the "benchmark" folder
pull                Pulls the official container tagged cpu-latest or gpu-latest from Docker
hub                 # TODO: ADD DESCRIPTION
test-container      Ensures that your locally built container can import all the Python packages
                    successfully when it runs
test-submission     Runs container with submission/submission.zip as your submission and inference-data
                    as the data to work with
```

To find out more about what these commands do, keep reading! :eyes:

## (1) Testing your submission.zip

### Implement your solution

In order to test your code submission, you will need a code submission! You will need to train your model separately before creating your `submission.zip` file that will perform inference.

**NOTE: You WILL implement all of your training and experiments on your machine. It is highly recommended that you use the same package versions as we do in the inference runtime definition ([cpu](runtime/py-cpu.yml) or [gpu](runtime/py-gpu.yml)). They can be installed with `conda`.**

The [submission format page](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/page/257/) contains the detailed information you need to prepare your submission.

### How your submission will run

Your submission will be unzipped into the working directory `/inference`. We will then run a Python process in that working directory to execute the `main.py` extracted from your submission. This `main.py` should read the `submission_format.csv` and `test_metadata.csv` files from `/inference/data`. The test images will also exist within the folder `/inference/data`. On the DrivenData platform, `/inference/data` will have the actual test images, and matching `submission_format.csv` and `test_metadata.csv`. Since you do not have the test set images, you should update the ones here to reference versions of the images from the training set. You should add the same test set images that appear here.

Running this command will download three images to that folder which match the metadata and can be used for testing:

```bash
make sample-images
```

When you execute the container locally, we will mount two subfolders in this repository into the containter:

- the `inference-data` directory is mounted in your locally running container as a read-only directory `/inference/data`
-  the `submission` directory is mounted in your locally running container as `/inference/submission`

Your `submission.zip` file must exist in the `submission` folder here in order to be processed when you are testing execution locally.

To prepare the example submission and put it into the submission folder, first check if you already have a file named `submission.zip` in the `submission` folder. If so, you'll need to remove it (we don't do it automatically so that you don't accidentally lose your work). Then run:

```bash
make pack-benchmark
```

### Test running your submission locally

You can execute the same containers locally that we will use on the DrivenData platform to ensure your code will run.

Make sure you have the [prequisites](#prequisites) installed. Then, you can run the following command within the repository to download the official image:

```bash
make pull
```

### Making a submission

Once you have the container image downloaded locally, you will be able to run it to see if your inference code works. You can put your `submission.zip` file in the `submission` folder and run the following command (or just use the sample one that was created when you ran `make pack-benchmark` above):

```bash
make test-submission
```

This will spin up the container, mount the local folders as drives within the folder, and follow the same steps that will run on the platform to unpack your submission and run inference against what it finds in the `/inference/data` folder.

### Reviewing the logs

When you run `make test-submission` the logs will be printed to the terminal. They will also be written to the `submission` folder as `log.txt`. You can always review that file and copy any versions of it that you want from the `submission` folder. The errors there will help you to determine what changes you need to make so your code executes successfully.

## (2) Updating the runtime packages

We accept contributions to add dependencies to the runtime environment. To do so, follow these steps:

1. Fork this repository
2. Make your changes
3. Test them and commit using git
3. Open a pull request to this repository

If you're new to the GitHub contribution workflow, check out [this guide by GitHub](https://guides.github.com/activities/forking/).

### Adding dependencies to the runtime

We use [conda](https://docs.conda.io/en/latest/) to manage Python dependencies. Add your new dependencies to both `runtime/py-cpu.yml` and `runtime/py-gpu.yml`. Please also add your dependencies to `runtime/tests/test-installs.py`, below the line `## ADD ADDITIONAL REQUIREMENTS BELOW HERE ##`.

Your new dependency should follow the format in the yml and be pinned to a particular version of the package and build with conda.

### Testing new dependencies

Please test your new dependency locally by recreating the relevant conda environment using the appropriate CPU or GPU `.yml` file. Try activating that environment and loading your new dependency.

Once that works, you'll want to make sure it works within the container as well. To do so, you can run (note this will run `make build` to create the new container image automatically, but you could also do it manually):

```
make test-container
```

This will build a local version of the official container and then run the import tests to make sure the relevant libraries can all be successfully loaded. This must pass before you submit a pull request to our repo to update the requirements. If it does not, you'll want to figure out what else you need to make the dependencies happy.

This command will run a bash shell in the container to let you interact with it. Make sure to activate the `conda` environment if you want to test the dependencies!

```
make debug-container
```

### Opening a pull request

After making and testing your changes, commit your changes and push to your fork. Then, when viewing the repository on github.com, you will see a banner that lets you open the pull request. For more detailed instructions, check out [GitHub's help page](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

Once you open the pull request, Github Actions will automatically try building the Docker images with your changes and run the tests in `runtime/tests`. These tests take ~30 minutes to run through, and may take longer if your build is queued behind others. You will see a section on the pull request page that shows the status of the tests and links to the logs.

You may be asked to submit revisions to your pull request if the tests fail, or if a DrivenData team member asks for revisions. Pull requests won't be merged until all tests pass and the team has reviewed and approved the changes.

---

## Happy modeling!

Thanks for reading! Enjoy the competition, and [hit up the forums](https://community.drivendata.org/) if you have any questions!
