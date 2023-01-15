from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://sellcar.netlify.app",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:8000",
    "https://car-preds-price.herokuapp.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('pipe.pkl', 'rb') as f:
    model = pickle.load(f)

class PriceItem(BaseModel):
    Location:str
    Year:int
    Kilometers_Driven:int
    Fuel_Type:str
    Transmission:str
    Owner_Type:str
    Seats:int
    Company:str
    Mileage_km_per_kg:float
    Engine_cc:float
    Power_bhp:float

@app.post('/')
async def price_endpoint(item:PriceItem):

    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    yhat = model.predict(df)
    print(yhat)
    return {'Predcition': np.round(float(yhat),2)}

# if __name__ == "__main__":
#     uvicorn.run("main:app", port=5000)
