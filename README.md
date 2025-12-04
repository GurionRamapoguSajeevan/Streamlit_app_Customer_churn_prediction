# 📊 Telco Customer Churn Prediction Streamlit App

An interactive web application that predicts customer churn using a trained machine learning model. This app provides instant churn risk predictions and makes model insights easily accessible through a clean, user-friendly interface.

# View the web app here: 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churnprediction-app-grs-wqyhm9zikea2uleappszlmc.streamlit.app/)

# Overview
* This Streamlit application is built to demonstrate real-time churn prediction using a production-ready machine learning pipeline.
* It loads a pre-trained model, processes user inputs, and generates clear predictions that help telecom business stakeholders assess customer churn risk.

This repository is optimized specifically for deployment (e.g., Streamlit Cloud).

# Key Features
* Real-time churn prediction
* Fast, lightweight UI built with Streamlit
* Uses a pre-trained machine learning model (included as .pkl)
* Automatic preprocessing within the app (no manual steps)
* Clean input controls for customer attributes
* Works out-of-the-box with minimal setup

# 📁 Repository Structure

├── model_outputs.pkl                      # Pre-trained model (.pkl) and artifacts

├── streamlit_app_churn_prediction.py      

├── requirements.txt                       # Dependencies for deployment

├── .gitattributes                         # LFS settings for large model files `(model_outputs.pkl)`

├── .gitignore                             # Ignore unnecessary files

├── LICENSE               

└── README.md

# ⚙️ Installation & Running Locally
Clone the repository:

`git clone https://github.com/GurionRamapoguSajeevan/Streamlit_app_Customer_churn_prediction.git`

`cd Streamlit_app_Customer_churn_prediction`

# Install dependencies:

`pip install -r requirements.txt`

# Run the Streamlit app locally:

`streamlit run streamlit_app_churn_prediction.py`

# App Deployment
This repository is structured for seamless deployment on:
* Streamlit Cloud
* Heroku (with minimal adjustments)
* Any container-based hosting service

# Please make sure the .pkl model file is tracked via _Git LFS_ due to its size.

# Author: _Gurion_

[_LinkedIN_](https://www.linkedin.com/in/rs-gurion/)

[_Complete Churn Prediction ML & Analytics Project_](https://github.com/GurionRamapoguSajeevan/Customer-Churn-Predictive-Analytics-Suite)

If you have questions, suggestions, or want to collaborate, feel free to reach out via GitHub @GurionRamapoguSajeevan.
