# Immo_MLOps

Machine learning project for real estate price prediction in France.</br>
The goal of this project is to build a simple MLOps pipeline that collects real estate data, preprocesses it, trains a model, evaluates its performance, and serves predictions through an API.

## Project structure

Immo_MLOps/
 │ ├── api.py
   ├── requirements.txt 
   ├── src/ 
   │ ├── collect.py 
   │ ├── process.py 
   │ ├── train.py 
   │ ├── predict.py 
   │ ├── evaluate.py 
   │ └── fake_new_data.py 
   │ 
   ├── dags/ 
   │     └── training_pipeline.py 
   │ 
   ├── data/ 
   └── models/