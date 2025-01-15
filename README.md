# Master Thesis
### Vessel Fuel Consumption Prediction

This project provides a web application that predicts vessel fuel consumption per nautical mile based on input features such as the vessel's year built, main engine KW, deadweight, and gross rated tonnage (GRT). The application uses machine learning models (linear regression and neural networks) to estimate fuel consumption, and visualizes the results with interactive plots.

---
### Requirements

- Python 3.6+
- Dash
- NumPy
- Pandas
- Plotly
- Keras
- Scikit-learn
- Joblib
- Pickle

To install the required dependencies, use the following:

pip install -r requirements.txt

### Application

An application of fuel consumption prediction will be localy run in http://127.0.0.1:8050/

Vessel features:
 - **Vessel's Yearbuilt**: The year the vessel was built (between 1990 and 2022).
 - **Vessel's Main Engine KW:** The power of the vessel’s main engine in kilowatts (from 1,000 to 40,000 KW).
 - **Vessel's Deadweight:** The vessel’s deadweight tonnage (from 6,000 to 350,000 tons).
 - **Vessel's Gross Rated Tonnage (GRT):** The gross rated tonnage of the vessel (from 4,500 to 180,000 tons).

### Models

You have two models running predictions on your fuel consumption predictions:
 - **Linear Model Projection**
 - **Custom Neural Network Projection**

### Graphs

 - Fuel consumption for vessels with the same year built: A scatter plot comparing deadweight and fuel consumption per nautical mile for vessels built in the selected year.
 - Fuel consumption for vessels with similar deadweight: A scatter plot comparing year built and fuel consumption per nautical mile for vessels with a deadweight within ±10% of the selected value.

---

 ![application](https://github.com/user-attachments/assets/cf9347b8-a468-4177-8d2b-b6cd8f23caa6)
 
 ![application_2](https://github.com/user-attachments/assets/e1ccc505-61f6-4cfd-b617-71c2437bc4f1)

  
