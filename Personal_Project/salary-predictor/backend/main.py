from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Salary Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'model.joblib'))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    explainer = joblib.load(os.path.join(MODEL_DIR, 'explainer.joblib'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
except Exception as e:
    print("Warning: Failed to load models. They will be needed for prediction.")
    print(e)
    model = None

class JobInput(BaseModel):
    YearsExperience: float
    Education: str
    Role: str
    Location: str

@app.post("/predict")
def predict_salary(job: JobInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
        
    input_df = pd.DataFrame([job.model_dump()])
    
    try:
        X_processed = preprocessor.transform(input_df).toarray().astype(float)
        prediction = model.predict(X_processed)[0]
        shap_values = explainer.shap_values(X_processed)
        
        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else explainer.expected_value
        
        shap_dict = []
        for i, name in enumerate(feature_names):
            shap_dict.append({
                "feature": name,
                "value": float(shap_values[0][i])
            })
            
        original_features = ["Education", "Role", "Location", "YearsExperience"]
        aggregated_shap = {feat: 0.0 for feat in original_features}
        
        for i, name in enumerate(feature_names):
            val = float(shap_values[0][i])
            if name.startswith('Education_'):
                aggregated_shap['Education'] += val
            elif name.startswith('Role_'):
                aggregated_shap['Role'] += val
            elif name.startswith('Location_'):
                aggregated_shap['Location'] += val
            elif name == 'YearsExperience':
                aggregated_shap['YearsExperience'] += val
                
        aggregated_shap_list = [{"feature": k, "value": v} for k, v in aggregated_shap.items()]
        aggregated_shap_list.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "prediction": float(prediction),
            "base_value": float(expected_value),
            "shap_values": aggregated_shap_list,
            "raw_shap_values": shap_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Salary Predictor API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
