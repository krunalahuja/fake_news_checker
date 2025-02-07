Fake News Detection Model with Flask
This repository contains a machine learning model for detecting fake news articles. The model is trained using a Decision Tree classifier and is integrated into a Flask web application for real-time predictions. The project includes text preprocessing, feature extraction using TF-IDF, and a user-friendly web interface.

Table of Contents
Overview

Features

Installation

Usage

Dataset

Model Training

Flask Application

Contributing


Overview
Fake news is a growing concern in today's digital age. This project aims to provide a tool for detecting fake news articles using machine learning. The model is trained on a dataset of real and fake news articles and is deployed as a web application using Flask.

Features
Text Preprocessing: Cleans and preprocesses text data by removing special characters, stopwords, and lemmatizing words.

TF-IDF Vectorization: Converts text into numerical features for machine learning.

Dataset
The model is trained on two datasets:

True.csv: Contains real news articles.

Fake.csv: Contains fake news articles.

These datasets are combined, preprocessed, and used to train the model.

Model Training
The model is trained using a Decision Tree classifier. Here's a summary of the steps:

Preprocessing: Text data is cleaned and tokenized.

Feature Extraction: TF-IDF vectorization is applied to convert text into numerical features.

Training: The model is trained on the preprocessed dataset.

Evaluation: The model's accuracy is calculated on a test set.

Decision Tree Classifier: A lightweight and interpretable model for fake news detection.

Flask Web Application: Provides a user-friendly interface for real-time predictions.

Scalable: The model and vectorizer are saved as files, making it easy to update or replace them.

Flask Application
The Flask app provides a simple web interface for interacting with the trained model. It includes:

A home page with a text input form.

A prediction route that processes the input and displays the result.


Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.

Create a new branch for your feature or bugfix.

Commit your changes and push to the branch.

Submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Dataset: Kaggle Fake and Real News Dataset

Libraries: pandas, nltk, scikit-learn, Flask
