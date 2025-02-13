import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

with open("winemodel.pkl", "rb") as f:
    model = pickle.load(f)

class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float          
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float 
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


@app.get("/")
async def read_root():
    return{"Hello": "World"}

@app.post("/predict")
def predict(wine: Wine):
    data = wine.dict()
    data_df = pd.DataFrame(data, index=[0])
    prediction = model.predict(data_df)
    return {
        "prediction": prediction[0]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)