# Pix2Struct
This repository contains code for [Pix2Struct: Screenshot Parsing as Pretraining
for Visual Language Understanding](https://arxiv.org/abs/2210.03347).

We release pretrained checkpoints for the Base and Large models and code for
finetuning them on the nine downstream tasks discussed in the paper.
We are unable to release the pretraining data, but they can be replicated using
the publicly available URLs released in the
[C4 dataset](https://www.tensorflow.org/datasets/catalog/c4).

# Getting Started
Clone the github repository, install the `pix2struct` package, and run
the tests to ensure that all dependencies were successfully installed.

```
git clone https://github.com/google-research/pix2struct.git
cd pix2struct
conda create -n pix2struct python=3.9
conda activate pix2struct
pip install -e ."[dev]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pytest
```

You may first need to install Java (`sudo apt install default-jre`) and
[conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
if not already installed.

We will be using Google Cloud Storage (GCS) for data and model storage. For the
remaining documentation we will assume that the path to your own bucket and
directory is in the `PIX2STRUCT_DIR` environment variable:

```
export PIX2STRUCT_DIR="gs://<your_bucket>/<path_to_pix2struct_dir>"
```

The code for running experiments assumes this environment variable when looking
for the preprocessed data.

# Data Preprocessing

Our data preprocessing scripts are run with [Dataflow](https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python)
by default using the [Apache Beam library](https://cloud.google.com/dataflow/docs/concepts/beam-programming-model).
They can also be run locally by turning off flags appearing after `--`.

For the remaining documentation we will assume that GCP project information is
in the following environment variables:

```
export GCP_PROJECT=<your_project_id>
export GCP_REGION=<your_region>
```

Below are the commands required to preprocess each dataset. The results will
be written to `$PIX2STRUCT_DIR/data/<task_name>/preprocessed/`, which is the
file structure assumed in `tasks.py`.

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

## ChartQA
```
mkdir -p data/chartqa
cd data/chartqa
git clone https://github.com/vis-nlp/ChartQA.git
cp -r ChartQA/ChartQA\ Dataset/* ./
rm -rf ChartQA
cd ..
gsutil -m cp -r chartqa $PIX2STRUCT_DIR/data/chartqa
python -m pix2struct.preprocessing.convert_chartqa \
  --data_dir=$PIX2STRUCT_DIR/data/chartqa \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

## RICO Images
Screen2Words, RefExp, and Widget Captioning all require images from the RICO
dataset. If you'd like to use any of these datasets, please process RICO images
before proceeding.

```
cd data
wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz
tar xvfz unique_uis.tar.gz
rm unique_uis.tar.gz
gsutil -m cp -r combined $PIX2STRUCT_DIR/data/rico_images
```

## Widget Captioning
If you haven't already setup RICO, please do so before you proceed.

```
mkdir -p data/widget_captioning
cd data/widget_captioning
git clone https://github.com/google-research-datasets/widget-caption.git
cp widget-caption/widget_captions.csv ./
cp widget-caption/split/*.txt ./
mv dev.txt val.txt
rm -rf widget-caption
cd ..
gsutil -m cp -r widget_captioning $PIX2STRUCT_DIR/data/widget_captioning
python -m pix2struct.preprocessing.convert_widget_captioning \
  --data_dir=$PIX2STRUCT_DIR/data/widget_captioning \
  --image_dir=$PIX2STRUCT_DIR/data/rico_images \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

## Screen2Words
If you haven't already setup RICO, please do so before you proceed.

```
cd data
git clone https://github.com/google-research-datasets/screen2words.git
gsutil -m cp -r screen2words $PIX2STRUCT_DIR/data/screen2words
python -m pix2struct.preprocessing.convert_screen2words \
  --screen2words_dir=$PIX2STRUCT_DIR/data/screen2words \
  --rico_dir=$PIX2STRUCT_DIR/data/rico_images \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

## RefExp
If you haven't already setup RICO, please do so before you proceed.

```
mkdir -p data/refexp
cd data/refexp
wget https://github.com/google-research-datasets/uibert/raw/main/ref_exp/train.tfrecord
wget https://github.com/google-research-datasets/uibert/raw/main/ref_exp/dev.tfrecord
wget https://github.com/google-research-datasets/uibert/raw/main/ref_exp/test.tfrecord
mv dev.tfrecord val.tfrecord
cd ..
gsutil -m cp -r refexp $PIX2STRUCT_DIR/data/refexp
python -m pix2struct.preprocessing.convert_refexp \
  --data_dir=$PIX2STRUCT_DIR/data/refexp \
  --image_dir=$PIX2STRUCT_DIR/data/rico_images \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

## DocVQA
```
mkdir -p data/docvqa
cd data/docvqa
```
Download DocVQA (Single Document Visual Question Answering) from
[the official source](https://rrc.cvc.uab.es/?ch=17&com=downloads) (requires
registration). The following steps assume that the train/val/test.tar.gz files
are in `data/docvqa`.

```
tar xvf train.tar.gz
tar xvf val.tar.gz
tar xvf test.tar.gz
rm -r *.tar.gz */ocr_results

cd ..
gsutil -m cp -r docvqa $PIX2STRUCT_DIR/data/docvqa
python -m pix2struct.preprocessing.convert_docvqa \
  --data_dir=$PIX2STRUCT_DIR/data/docvqa \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

## InfographicVQA
```
mkdir -p data/infographicvqa
cd data/infographicvqa
```
Download InfographicVQA Task 1 from [this](https://rrc.cvc.uab.es/?ch=17&com=downloads)
website (requires registration). The following steps assume that the
`train/val/test.json` and the `zip` files are in `data/infographicvqa`.

```
for split in train val test
do
  unzip infographicVQA_${split}_v1.0_images.zip
  mv infographicVQA_${split}_v1.0_images $split
  mv infographicVQA_${split}_v1.0.json $split/${split}_v1.0.json
done
rm *.zip

cd ..
gsutil -m cp -r infographicvqa $PIX2STRUCT_DIR/data/infographicvqa
python -m pix2struct.preprocessing.convert_docvqa \
  --data_dir=$PIX2STRUCT_DIR/data/infographicvqa \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

## OCR-VQA
```
mkdir -p data/ocrvqa
cd data/ocrvqa
```
Follow instructions on the [OCR-VQA](https://ocr-vqa.github.io/) website to
download the data into `data/ocrvqa` (requires crawling). The following steps
assume that `data/ocrvqa` contains a directory called `images` and a file called
`dataset.json`.

```
cd ..
gsutil -m cp -r ocrvqa $PIX2STRUCT_DIR/data/ocrvqa
python -m pix2struct.preprocessing.convert_ocrvqa \
  --data_dir=$PIX2STRUCT_DIR/data/ocrvqa \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

## AI2D
```
mkdir -p data/
cd data/
wget https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip
unzip ai2d-all.zip
rm ai2d-all.zip
gsutil -m cp -r ai2d $PIX2STRUCT_DIR/data/ai2d
python -m pix2struct.preprocessing.convert_ai2d \
  --data_dir=$PIX2STRUCT_DIR/data/ai2d \
  --test_ids_path=gs://pix2struct-data/ai2d_test_ids.csv \
  -- \
  --runner=DataflowRunner \
  --save_main_session \
  --project=$GCP_PROJECT \
  --region=$GCP_REGION \
  --temp_location=$PIX2STRUCT_DIR/data/temp \
  --staging_location=$PIX2STRUCT_DIR/data/staging \
  --setup_file=./setup.py
```

# Running experiments

The main experiments are implemented as a light wrapper around the
[T5X](https://github.com/google-research/t5x) library. For brevity, we
illustrate an example workflow of finetuning the pretrained base Pix2Struct
model on the Screen2Words dataset. To scale up to larger setups, please see
to the T5X documentation.

## Setting up the TPU

Following official [instructions](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)
for running JAX on a Cloud TPU VM, which allows you to directly `ssh` into the
TPU host.

In this example, we are using a `v3-8` TPU:

```
TPU_TYPE=v3-8
TPU_NAME=pix2struct-$TPU_TYPE
TPU_ZONE=europe-west4-a
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$TPU_ZONE \
  --accelerator-type=$TPU_TYPE \
  --version=tpu-vm-base
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE
```

Once you have `ssh`ed into the TPU host, follow the "Getting Started"
instructions to install the `pix2struct` package.

## Training
The following command will initiate the training loop, which consists of train
steps interleaved with evaluations on the validation set.

```
python -m t5x.train \
  --gin_search_paths="pix2struct/configs" \
  --gin_file="models/pix2struct.gin" \
  --gin_file="runs/train.gin" \
  --gin_file="sizes/base.gin" \
  --gin_file="optimizers/adafactor.gin" \
  --gin_file="schedules/screen2words.gin" \
  --gin_file="init/pix2struct_base_init.gin" \
  --gin.MIXTURE_OR_TASK_NAME="'screen2words'" \
  --gin.MODEL_DIR="'$PIX2STRUCT_DIR/experiments/screen2words_base'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 4096, 'targets': 128}" \
  --gin.BATCH_SIZE=32
```

## Evaluation
The following command evaluates the model on the test set. You will need to
replace the checkpoint path with the one that was actually selected based on the
validation performance.

```
python -m t5x.eval \
  --gin_search_paths="pix2struct/configs" \
  --gin_file="models/pix2struct.gin" \
  --gin_file="runs/eval.gin" \
  --gin_file="sizes/base.gin" \
  --gin.MIXTURE_OR_TASK_NAME="'screen2words'" \
  --gin.CHECKPOINT_PATH="'$PIX2STRUCT_DIR/experiments/screen2words_base/checkpoint_286600'" \
  --gin.EVAL_OUTPUT_DIR="'$PIX2STRUCT_DIR/experiments/test_exp/test_eval'" \
  --gin.EVAL_SPLIT="'test'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 4096, 'targets': 128}" \
  --gin.BATCH_SIZE=32
```

## Finetuned Checkpoints
In addition to the pretrained checkpoints released and specified in the
`configs/init` directory. We also release checkpoints for the finetuned models
on all tasks below.

| Task             | GCS Path (Base)                                               | GCS Path (Large)                                               |
| -----------------| ------------------------------------------------------------- | -------------------------------------------------------------- |
| TextCaps         | `gs://pix2struct-data/textcaps_base/checkpoint_280400`          | `gs://pix2struct-data/textcaps_large/checkpoint_180600`          |
| ChartQA          | `gs://pix2struct-data/chartqa_base/checkpoint_287600`           | `gs://pix2struct-data/charqa_large/checkpoint_182600`            |
| WidgetCaptioning | `gs://pix2struct-data/widget_captioning_base/checkpoint_281600` | `gs://pix2struct-data/widget_captioning_large/checkpoint_181600` |
| Screen2Words     | `gs://pix2struct-data/screen2words_base/checkpoint_282600`      | `gs://pix2struct-data/screen2words_large/checkpoint_183000`      |
| RefExp           | `gs://pix2struct-data/refexp_base/checkpoint_290000`            | `gs://pix2struct-data/refexp_large/checkpoint_187800`            |
| DocVQA           | `gs://pix2struct-data/docvqa_base/checkpoint_284400`            | `gs://pix2struct-data/docvqa_large/checkpoint_184000`            |
| InfographicVQA   | `gs://pix2struct-data/infographicvqa_base/checkpoint_284000`    | `gs://pix2struct-data/infographicvqa_large/checkpoint_182000`    |
| OCR-VQA          | `gs://pix2struct-data/ocrvqa_base/checkpoint_290000`            | `gs://pix2struct-data/ocrvqa_large/checkpoint_188400`           |
| AI2D             | `gs://pix2struct-data/ai2d_base/checkpoint_284400`              | `gs://pix2struct-data/ai2d_large/checkpoint_184000`              |

These checkpoints are compatible with the eval command documented above and the
two ways of performing inference mentioned below. Please ensure that the config
file under `configs/sizes` is set to be consistent with the checkpoint.


## Inference

We provide two ways of performing inference. For testing and demoing purposes,
these may be run on CPU. In that case, please set the `JAX_PLATFORMS`
environment variable to `cpu`.

### Command-line example

We provide a minimal script for performing inference on a single example. This
path has only been tested at extremely small scale and is not meant for
larger-scale inference. For large-scale inference, we recommend setting a custom
task with placeholder labels and running the evaluation script (`t5x.eval`) as
documented above.

In the following example, we show the command for predicting the caption of an
image using a base-sized checkpoint finetuned on the TextCaps task. For a task
that also accepts textual prompts such as questions in VQA, you can also supply
the question via the `text` flag (in addition to specifying the image with the
`image` flag).

```
python -m pix2struct.example_inference \
  --gin_search_paths="pix2struct/configs" \
  --gin_file=models/pix2struct.gin \
  --gin_file=runs/inference.gin \
  --gin_file=sizes/base.gin \
  --gin.MIXTURE_OR_TASK_NAME="'placeholder_pix2struct'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 2048, 'targets': 128}" \
  --gin.BATCH_SIZE=1 \
  --gin.CHECKPOINT_PATH="'gs://pix2struct-data/textcaps_base/checkpoint_280400'" \
  --image=$HOME/test_image.jpg
```

### Web Demo

For a more user-friendly demo, we also provide a web-based alternative of
inference script above. While running this command, the web demo can be accessed
at `localhost:8080` (or any port specified via the `port` flag), assuming you
are running the demo locally. You can then upload your custom image and optional
prompt instead of specifying it via the command line.

```
python -m pix2struct.demo \
  --gin_search_paths="pix2struct/configs" \
  --gin_file=models/pix2struct.gin \
  --gin_file=runs/inference.gin \
  --gin_file=sizes/base.gin \
  --gin.MIXTURE_OR_TASK_NAME="'placeholder_pix2struct'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 2048, 'targets': 128}" \
  --gin.BATCH_SIZE=1 \
  --gin.CHECKPOINT_PATH="'gs://pix2struct-data/textcaps_base/checkpoint_280400'"
```

## Clean up
When you are done with your TPU VM, remember to delete the instance:

```
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$TPU_ZONE
```

# Note

*This is not an officially supported Google product.*
