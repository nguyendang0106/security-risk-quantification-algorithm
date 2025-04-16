from fastapi import FastAPI
import pickle

with open("XGB_attack_all.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ['Generic', 'Shellcode', 'Exploits', 'Reconnaissance', 'Backdoor', 
               'Normal', 'Analysis', 'Fuzzers', 'DoS', 'Worms']

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ML model deployment"}

@app.post("/predict")
def predict(data: dict):
    """
    Predict the class of the input data using the loaded model.
    """
    # Preprocess the input data
    data = [data['data']]
    
    # Make prediction
    prediction = model.predict(data)
    
    # Map prediction to class name
    class_name = class_names[prediction[0]]
    
    return {"prediction": class_name}