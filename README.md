#  Modeling Metastasis
### Predicting cancer outcomes in Merkel Cell Carcinoma to improve clinical care.


Consulting project for Insight Health Data Science
with the Cutaneous Research Lab, Dr. Wesley Yu at OHSU 

## Project description 

**Modeling:** This project applies machine learning to predict risk of metastasis in Merkel Cell Carcinoma patients. 
The dataset consists of 1K patients with MCC diagnoses from the National Cancer Database (NCDB). 
The features of interest include tumor characteristics and patient demographics. 
The outcome of interest is positive case of metastasis, designated from sentinel lymph node biopsy results. 

**Web application:** Clinical prototype to guide decision-making for physicians and patients about whether to conduct invasive lymph node biopsy. 
The app deploys the ML model and provides estimated probability of metastasis based on input data from a new  patient. 
Currently deployed using streamlit's github sharing service [here](healthpredictions.xyz).

[Slides](https://docs.google.com/presentation/d/1-1j8M5oHO6jLQpZEecjuQYNstHOt49aAP7QeE8-4N6s/edit#slide=id.ga1e0b50080_0_0)   |   [Demo](https://youtu.be/o4iRkPfRkaA)





## Contents 
**notebooks:** Jupyter notebooks to walk through data cleaning, EDA, and machine learning pipeline.

**app:** live_app.py contains the source code for Streamlit app, which deploys the machine learning model and provides results from new data. 
Requirements.txt provides the required packages to deploy the streamlit app via their github sharing service.

**mcc_metastasis:** Custom functions, setup and configuration for the project.

**figures:** Figures generated to display EDA and model results.

**model output:** Storing final model deployed in the application and associated model results. 

**data:** Contains raw, cleaned and processed (imputed) datasets for analysis, in addition to the data dictionary from NCDB.

