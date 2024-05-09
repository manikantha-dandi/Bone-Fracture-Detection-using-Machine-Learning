
# Bone Fracture Detection using Machine Learning

This repository contains all the necessary files for deploying a machine learning model that identifies bone fractures from X-ray images. The model utilizes Random Forest algorithm for its robustness and accuracy in handling such classifications.

## Repository Contents

- `Dataset_Link.txt` - Contains the link to the dataset used for training the model. The dataset includes images of arms and legs X-rays labeled as fractured or non-fractured.
- `Evaluation File.ipynb` - A Jupyter notebook that details the model's evaluation process including performance metrics such as precision, recall, accuracy, F1 score, and AUC score.
- `app.py` - The Python script for the web application that utilizes Flask to serve the trained model. This application allows users to upload X-ray images and receive predictions.
- `random_forest_model_nand.pkl` - The serialized form of the trained Random Forest model ready to be loaded and used for predictions.
