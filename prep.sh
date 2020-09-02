#!/bin/bash
set -e

# which language; default py for python
lang=py

# clean existing submission
if [ -f submission/submission.zip ]; then
    read -p "submission/submission.zip exists already. Remove? [y/n] : " yn
    case $yn in
        [Yy]* ) rm submission/submission.zip;;
        [Nn]* ) echo "... not continuing submission prep."; exit;;
        * ) echo "Please answer y or n.";;
    esac
fi

# prepare submission
cd benchmark/inference-$lang/; zip -r ../../submission/submission.zip ./*; cd ../..