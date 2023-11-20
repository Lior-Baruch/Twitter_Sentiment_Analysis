# Twitter Sentiment Analysis

## Overview
This project is focused on developing a Twitter Sentiment Analysis model. It leverages natural language processing (NLP) techniques to classify tweets into various sentiment categories.

## Data
The dataset comprises of tweets that are categorized into different sentiments. The data files are located under the `Data/` directory with the following structure:
- `twitter_training.csv`: Contains training data.
- `twitter_validation.csv`: Contains validation data.

## Installation
To run this notebook, ensure you have the following Python libraries installed:
- Pandas
- Matplotlib
- Seaborn
- Torch
- Transformers

You can install them using pip:
```bash
pip install pandas matplotlib seaborn torch transformers
```

## Usage
The notebook is divided into several sections, each handling a different aspect of the sentiment analysis process.

1. **Data Loading**: Load and display basic information about the dataset.
2. **Exploratory Data Analysis**: Visualize the distribution of data.
3. **Data Preprocessing**: Clean and preprocess the data for model training.
4. **Model Setup**: Define and initialize the BERT model for sequence classification.
5. **Training Preparation**: Set up the training loop, including loss functions and optimizers.
6. **Training and Validation**: Train the model on the training dataset and validate it.
7. **Evaluation and Plotting**: Plot training and validation loss and accuracy.
8. **Saving Model and History**: Save the trained model and training history.
9. **Testing with Custom Inputs**: Test the model with user-defined text inputs.

## Model
The model used is a BERT model for sequence classification, fine-tuned on the provided Twitter dataset.

## File Structure
- `Models/`: Directory to store trained models and their histories.
- `Data/`: Directory containing the training and validation datasets.

## Note
The code is still under development, and some sections might be updated or modified in future iterations.
