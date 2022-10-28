# pix2struct

# Getting Started

```
conda create -n pix2struct python=3.9
conda activate pix2struct
pip install -e .[dev]
pytest
```

We will be using Google Cloud Storage (GCS) for data and model storage. For the
remaining documentation we will assume that the path to your own bucket and
directory is in the `PIX2STRUCT_DIR` environment variable:

```
export PIX2STRUCT_DIR="gs://<your_bucket>/<path_to_pix2struct_dir>"
```

# Data Preprocessing

Our data preprocessing scripts are run with [Dataflow] (https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python)
by default using the [Apache Beam library] (https://cloud.google.com/dataflow/docs/concepts/beam-programming-model).
They can also be run locally by turning off flags appearing after `--`.

For the remaining documentation we will assume that GCP project information is
in the following environment variables:
```
export GCP_PROJECT=<your_project>
export GCP_REGION=<your_region>
```

## TextCaps
```
mkdir -p data/textcaps
cd data/textcaps
curl -O https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json
curl -O https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json
curl -O https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_test.json
curl -O https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
curl -O https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip
unzip train_val_images.zip
rm train_val_images.zip
unzip test_images.zip
rm test_images.zip
cd ..
gsutil -m cp -r textcaps_data $PIX2STRUCT_DIR/data/textcaps
python -m pix2struct.preprocessing.convert_textcaps \
  --textcaps_dir=$PIX2STRUCT_DIR/data/textcaps \
  --output_dir=$PIX2STRUCT_DIR/data/textcaps/processed \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

# Note

*This is not an officially supported Google product.*
