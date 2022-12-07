# ViT vs CNN models for OCT Classification

> **Abstract:** 
*Retinal diseases like Diabetic Macular Edema (DME) and Drusen can lead to permanent blindness if not diagnosed correctly and timely. Non-invasive imaging
techniques, like OCT, are used for the diagnosis of these diseases. Deep learning methods have shown promising results in medical imaging-based diagnosis. CNN for instance, proved to perform very well in many image classification tasks. In addition, Vision Transformers (ViT) have also shown superior results in medical imaging classification. In this project, we aim to perform a comparative study of ViT and CNN-based models on the classification of a publicly available OCT dataset. Moreover, the effect of class imbalance has been investigated and addressed through data augmentation techniques and weighted sampling. Results suggest that CNN outperforms ViT when we have a limited dataset available while ViT can give better results through pre-training.* 

## Contents
1) [Usage](#usage) 
2) [Training](#training)
3) [Testing](#testing)

## Requirements
```bash
pip install -r requirements.txt
```
## Dataset
We used a publicly available OCT dataset, available [here](https://data.mendeley.com/datasets/rscbjbr9sj). This dataset contains 83484 2D OCT images for four different categories shown in figure 4a, namely Diabetic Macular Edema (DME), Choroidal Neovascularization (CNV), Drusen and a Normal (healthy) class. The dataset is divided into train and test sets. Table 4b shows the detailed count of the number of images in each class.

## Usage
Use the given command with the name of the model to train. 
List of models:
1) CNN (cnn.py)
2) Resnet50 (resnet50.py)
3) MobilenetV2 (mobilenet_v2.py)
4) Xception (xception.py)
5) ViT (vit.py)
6) Imagnet pretrained ViT (vit_imagenet.py)


## Training
<sup>([top](#contents))</sup>
Use the given command with the name of the model to train

```bash
python cnn.py \
  --mode train \
  --epochs 30 \
  --batch_size 64  \
  --learning_rate 0.0003 \
``` 

## Testing
<sup>([top](#contents))</sup>
Use the given command with the name of the model to train

```bash
python cnn.py \
  --mode test \
  --epochs 30 \
  --batch_size 64  \
  --learning_rate 0.0003 \
``` 

```
