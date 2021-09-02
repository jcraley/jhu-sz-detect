#!/bin/bash

# Create chb_01 EDF folder and download
mkdir -p EDF/chb01
wget -cO - https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf\?download > EDF/chb01/chb01_03.edf
wget -cO - https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf\?download > EDF/chb01/chb01_04.edf

# Create Manifest folder and create manifest
mkdir -p Manifests/chb01
echo "fn;fs;duration;nsamples;nchns;nsz;sz_starts;sz_ends;pt_num;onset_zone" > Manifests/chb01/chb01-test.csv
echo "fn;fs;duration;nsamples;nchns;nsz;sz_starts;sz_ends;pt_num;onset_zone" > Manifests/chb01/chb01-train.csv
echo "chb01_03.edf;256;3600;921600;18;0;[2996];[3036];1;-1" >> Manifests/chb01/chb01-test.csv
echo "chb01_04.edf;256;3600;921600;18;0;[1467];[1494];1;-1" >> Manifests/chb01/chb01-train.csv

# Preprocess the raw EDF files
python scripts/filter-and-normalize.py \
    --train_manifest Manifests/chb01/chb01-test.csv \
    --dataset chb01 --channel_list examples/chb-channels.txt
python scripts/filter-and-normalize.py \
    --train_manifest Manifests/chb01/chb01-train.csv \
    --dataset chb01 --channel_list examples/chb-channels.txt

# Extract features from the buffers
python scripts/extract-features.py \
    --train_manifest Manifests/chb01/chb01-test.csv \
    --dataset chb01 --channel_list examples/chb-channels.txt \
    --features '["bandpass"]'
python scripts/extract-features.py \
    --train_manifest Manifests/chb01/chb01-train.csv \
    --dataset chb01 --channel_list examples/chb-channels.txt \
    --features '["bandpass"]'

# Perform an experiment
python run_experiment.py \
    --train_manifest Manifests/chb01/chb01-train.csv \
    --val_manifest Manifests/chb01/chb01-test.csv \
    --dataset chb01 --channel_list examples/chb-channels.txt \
    --features '["bandpass"]' \
    --model_type RandomForest