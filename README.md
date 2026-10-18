# Garbage Classification with ResNet50 (Transfer Learning)

Classifies waste images into 6 classes: cardboard, glass, metal, paper, plastic and trash.

## Results
- Validation accuracy: about 94% with transfer learning, against about 25% for ResNet50 trained from scratch (10-epoch test run)
- Weakest class: trash (f1-score 0.81), because it is the smallest class
- Web app built with Gradio for uploading photos and getting a prediction

## What I changed
- Ran the project in Google Colab, loading the dataset from Google Drive
- Fixed a class-weights ordering bug (the weights were attached to the wrong classes)
- Added a Gradio web app

## Dataset
Trash Type Image Dataset (Kaggle): [Kaggle dataset][(PASTE DATASET LINK HERE)](https://www.kaggle.com/code/farzadnekouei/imbalanced-garbage-classification-resnet50)

## Credits
This project is based on the notebook "Imbalanced Garbage Classification | ResNet50" by Farzad Nekouei: [Kaggle notebook](PASTE KAGGLE NOTEBOOK LINK HERE)

Original code on GitHub: [Garbage_Classification_ResNet50_Scratch_to_Transfer-Learning](https://github.com/FarzadNekouee/Garbage_Classification_ResNet50_Scratch_to_Transfer-Learning)

## Notebook
`trash_classification_colab.ipynb`
