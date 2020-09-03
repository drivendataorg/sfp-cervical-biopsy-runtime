.PHONY: build pull test-container debug-container test-submission

# ================================================================================================
# Settings
# ================================================================================================

_NVIDIA = $(which nvidia-smi)
ifdef (_NVIDIA)
	CPU_OR_GPU = gpu
	GPU_ARGS = --gpus all
else
	CPU_OR_GPU = cpu
endif

REPO = drivendata/sfp-competition

TAG = ${CPU_OR_GPU}-latest
LOCAL_TAG = ${CPU_OR_GPU}-local

IMAGE = ${REPO}:${TAG}
LOCAL_IMAGE = ${REPO}:${LOCAL_TAG}

# To run a submission, wse local version if that exists; otherwise, use official version
SUBMISSION_IMAGE ?= $(shell docker images -q ${LOCAL_IMAGE})
SUBMISSION_IMAGE ?= $(shell docker images -q ${IMAGE})


# ================================================================================================
# Commands for building the container if you are changing the requirements
# ================================================================================================

## Builds the container locally, tagging it with cpu-local or gpu-local
build:
	docker build --build-arg CPU_GPU=${CPU_OR_GPU} -t ${LOCAL_IMAGE} runtime

## Ensures that your locally built container can import all the Python packages successfully when it runs
test-container: build
	docker run \
		${GPU_ARGS} \
		--mount type=bind,source=$(shell pwd)/runtime/run-tests.sh,target=/run-tests.sh,readonly \
		--mount type=bind,source=$(shell pwd)/runtime/tests,target=/tests,readonly \
		${LOCAL_IMAGE} \
		/bin/bash -c "bash /run-tests.sh ${CPU_OR_GPU}"

## Start your locally built container and open a bash shell within the running container; same as submission setup except has network access
debug-container: build
	docker run \
		${GPU_ARGS} \
		--mount type=bind,source=$(shell pwd)/inference-data,target=/inference/data,readonly \
		--mount type=bind,source=$(shell pwd)/submission,target=/inference/submission \
		--shm-size 8g \
		-it \
		${LOCAL_IMAGE} \
		/bin/bash


# ================================================================================================
# Commands for testing that your submission.zip will execute
# ================================================================================================

## Pulls the official container tagged cpu-latest or gpu-latest from Docker hub
pull:
	docker pull ${IMAGE}

## Creates a submission/submission.zip file from whatever is in the "benchmark" folder
pack-benchmark:
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif

	cd benchmark; zip -r ../submission/submission.zip ./*


## Runs container with submission/submission.zip as your submission and inference-data as the data to work with
test-submission:

# if container does not exists, error and tell user to pull or build
ifeq (,$(wildcard ./submission/submission.zip))
	$(error To test your submission, you must first put a "submission.zip" file in the "submission" folder. \
	  If you want to use the benchmark, you can run `make pack-benchmark` first)
endif

# if container does not exists, error and tell user to pull or build
ifeq (${SUBMISSION_IMAGE},)
	$(error To test your submission, you must first run `make pull` (to get official container) or `make build` \
		(to build a local version if you have changes).)
endif

	docker run \
		${GPU_ARGS} \
		--network none \
		--mount type=bind,source=$(shell pwd)/inference-data,target=/inference/data,readonly \
		--mount type=bind,source=$(shell pwd)/submission,target=/inference/submission \
	   	--shm-size 8g \
		${SUBMISSION_IMAGE}


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo
	@echo "$$(tput bold)Settings based on your machine:$$(tput sgr0)"
	@echo CPU_OR_GPU=${CPU_OR_GPU}  "\t\t\t# Whether or not to try to build, download, and run GPU versions"
	@echo SUBMISSION_IMAGE=${SUBMISSION_IMAGE}  "\t# ID of the image that will be used when running test-submission"
	@echo
	@echo "$$(tput bold)Available competition images:$$(tput sgr0)"
	@echo "$(shell docker images --format '{{.Repository}}:{{.Tag}}({{.ID}}}), ' ${REPO})"
	@echo
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
