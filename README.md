# MID-level-activations
Code for the paper [Towards Utilising a Range of Neural Activations for Comprehending Representational Associations](https://arxiv.org/abs/2411.10019). 

## Set up 

Clone the project and create and activate venv, or conda environment as desired (Tested on python3.9). 

```bash
https://github.com/lauraaisling/MID-level-activations.git
cd MID-level-activations
python3.9 -m venv .MID 
source .MID/bin/activate
pip install -r requirements.txt
pip install -e .
``` 

### Datasets and code

#### dSprites (for synthetic data demonstration)

From [here](https://github.com/deepmind/dsprites-dataset). 
Set the data path as appropriate (for later), and download the data. 

```bash
export LIB_DATA=$HOME/MID-level-activations/data

bash scripts/download_dSprites.sh
```

#### CelebA (for MID experiments)

Instructions summarised here: 

You can the dataset files from Kaggle at this [link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) ([Kaggle Datasets setup tutorial](https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/)). For reference, the original dataset [Liu et al. (2015)] can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 

```bash
kaggle datasets download jessicali9530/celeba-dataset
mkdir data/celebA
unzip celeba-dataset.zip -d data/celebA
rm -r celeba-dataset.zip
```

Then copy the files/celeba_metadata.csv file from the files folder as metadata.csv. 

```bash
cp files/celebA_metadata.csv data/celebA/metadata.csv
```

#### Waterbirds (for MID experiments)

Instructions from [here](https://github.com/kohpangwei/group_DRO#waterbirds) summarised below: 

You can download a [tarball](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) of this dataset as directed below. The Waterbirds dataset can also be accessed through the WILDS package, which will automatically download the dataset.

```bash
mkdir data/cub
wget -P data/cub https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xvzf data/cub/waterbird_complete95_forest2water2.tar.gz
```

### Log into wandb (else specify not using) and you're good to go!

```bash
wandb login
```

## Experiments

### dSprites 

```bash
jupyter notebook --no-browser --port=8080 & disown
```

ssh -L localhost:8080:localhost:8080 laura@patten.csis.ul.ie







## Step 1: Train ERM models

```bash
# Waterbirds
python src/spurious_feature_learning/train_supervised.py \
--output_dir logs/waterbirds/ERM_seed0 \
--model imagenet_resnet50_pretrained --seed 0 \
--num_epochs 100 --eval_freq 1 --save_freq 100 \
--weight_decay 1e-4 --batch_size 32 --init_lr 3e-3 \
--scheduler cosine_lr_scheduler \
--data_dir data/cub/waterbird_complete95_forest2water2 --dataset SpuriousCorrelationDataset \
--data_transform AugWaterbirdsCelebATransform \
--wandb_project activation-levels --wandb_name waterbirds_ERM0
# --wandb_track 

# CelebA
python src/spurious_feature_learning/train_supervised.py \
--output_dir logs/celebA/ERM_seed0_50ep \
--model imagenet_resnet50_pretrained --seed 0 \
--num_epochs 50 --eval_freq 1 --save_freq 100 \
--weight_decay 1e-4 --batch_size 100 --init_lr 1e-3 \
--scheduler cosine_lr_scheduler \
--data_dir data/celebA --dataset SpuriousCorrelationDataset \
--data_transform AugWaterbirdsCelebATransform \
--wandb_project activation-levels --wandb_name celebA_ERM0_50ep

# MultiNLI

python3 src/spurious_feature_learning/train_supervised.py \
--output_dir logs/multinli/ERM_seed0 \
--model bert_pretrained --seed 0 \
--num_epochs 10 --eval_freq 1 --save_freq 10 \
--weight_decay 1.e-4 --batch_size 16 --init_lr 1e-5 \
--scheduler bert_lr_scheduler --optimizer bert_adamw_optimizer \
--data_dir data/glue_data/MNLI --dataset MultiNLIDataset --data_transform None \
--wandb_project activation-levels --wandb_name MultiNLI_ERM0

```

## Step 2: Calculate embeddings and logits. DFR baseline also

```bash
# Waterbirds
python src/spurious_feature_learning/embeddings_evaluate_spurious.py \
--data_dir data/cub/waterbird_complete95_forest2water2 --dataset SpuriousCorrelationDataset \
--data_transform AugWaterbirdsCelebATransform --dataset_name waterbirds \
--model imagenet_resnet50_pretrained --ckpt_path logs/waterbirds/ERM_seed0/final_checkpoint.pt \
--result_path logs/waterbirds/ERM_seed0/wb_erm0final_embeddings_orig.pkl 

## DFR
python src/spurious_feature_learning/embeddings_evaluate_spurious.py \
--data_dir data/cub/waterbird_complete95_forest2water2 --dataset SpuriousCorrelationDataset \
--data_transform AugWaterbirdsCelebATransform --dataset_name waterbirds \
--model imagenet_resnet50_pretrained --ckpt_path logs/waterbirds/ERM_seed0/final_checkpoint.pt \
--mode DFR \
--result_path logs/waterbirds/ERM_seed0/wb_erm0final_DFR.pkl 

# CelebA
python src/spurious_feature_learning/embeddings_evaluate_spurious.py \
--data_dir data/celebA --dataset SpuriousCorrelationDataset \
--data_transform AugWaterbirdsCelebATransform --dataset_name celebA \
--model imagenet_resnet50_pretrained --ckpt_path logs/celebA/ERM_seed0/final_checkpoint.pt \
--result_path logs/celebA/ERM_seed0/celebA_erm0final_embeddings_orig.pkl 

## DFR
python src/spurious_feature_learning/embeddings_evaluate_spurious.py \
--data_dir data/celebA --dataset SpuriousCorrelationDataset \
--data_transform AugWaterbirdsCelebATransform --dataset_name celebA \
--model imagenet_resnet50_pretrained --ckpt_path logs/celebA/ERM_seed0/final_checkpoint.pt \
--mode DFR \
--result_path logs/celebA/ERM_seed0/celebA_erm0final_DFR.pkl 

```

## Step 3: Jupyter notebook

## Step 4: Retrain with MIDs

```bash
# Waterbirds
## MIDS
python src/spurious_feature_learning/embeddings_evaluate_spurious.py \
--data_dir data/cub/waterbird_complete95_forest2water2 --dataset SpuriousCorrelationDataset \
--data_transform AugWaterbirdsCelebATransform --dataset_name waterbirds \
--model imagenet_resnet50_pretrained --ckpt_path logs/waterbirds/ERM_seed0/final_checkpoint.pt \
--mode MIDS --mids_indices_mid_file results/waterbirds_erm0_all_mids_unique_minority_idxs_minority_idxs.txt \
--mids_indices_not_mid_file results/waterbirds_erm0_indices_not_mid_unique.txt \
--result_path logs/waterbirds/ERM_seed0/wb_erm0final_all_MIDS.pkl 
# --mids_indices_file results/waterbirds_all_mids_minority_idxs.txt \

# python src/spurious_feature_learning/embeddings_evaluate_spurious.py \
# --data_dir data/cub/waterbird_complete95_forest2water2 --dataset SpuriousCorrelationDataset \
# --data_transform AugWaterbirdsCelebATransform --dataset_name waterbirds \
# --model imagenet_resnet50_pretrained --ckpt_path logs/waterbirds/ERM_seed0/final_checkpoint.pt \
# --mode MIDS --mids_indices_file results/waterbirds_compart_mids_minority_idxs.txt \
# --result_path logs/waterbirds/ERM_seed0/wb_erm0final_compart_MIDS.pkl 

# CelebA
## MIDS
python src/spurious_feature_learning/embeddings_evaluate_spurious.py \
--data_dir data/celebA --dataset SpuriousCorrelationDataset \
--data_transform AugWaterbirdsCelebATransform --dataset_name celebA \
--model imagenet_resnet50_pretrained --ckpt_path logs/celebA/ERM_seed0/final_checkpoint.pt \
--mode MIDS --mids_indices_mid_file results/celebA_erm0_all_mids_unique_lab_eq_blip_minority_idxs.txt \
--mids_indices_not_mid_file results/celebA_erm0_indices_not_mid_unique.txt \
--result_path logs/celebA/ERM_seed0/celebA_erm0final_lab_eq_blip_minority_MIDS.pkl

# python src/spurious_feature_learning/embeddings_evaluate_spurious.py \
# --data_dir data/celebA --dataset SpuriousCorrelationDataset \
# --data_transform AugWaterbirdsCelebATransform --dataset_name celebA \
# --model imagenet_resnet50_pretrained --ckpt_path logs/celebA/ERM_seed0/final_checkpoint.pt \
# --mode MIDS --mids_indices_file results/celebA_compart_mids_minority_idxs.txt \
# --result_path logs/celebA/ERM_seed0/celebA_erm0final_compart_MIDS.pkl 

# MultiNLI
# python3 src/spurious_feature_learning/embeddings_evaluate_spurious.py \
# --data_dir data/glue_data/MNLI --dataset MultiNLIDataset \
# --data_transform None \
# --model bert_pretrained --ckpt_path logs/multinli/ERM_seed0/final_checkpoint.pt \
# --result_path logs/multinli/ERM_seed0/multinli_erm_seed0_dfr.pkl
```
