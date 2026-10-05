
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

MODEL_PATH = "/content/bt_land_ml_api/bt_land_price_model.pkl"

# Load trained ML model
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="BT Land Registry - Property Price Prediction API",
    description="ML service for property price estimation and price verification",
    version="1.1.0"
)

# 90th percentile model-error thresholds for verification
CITY_THRESHOLDS = {
    "Ahmedabad": 65.82,
    "Bangalore": 55.33,
    "Chandigarh": 48.72,
    "Chennai": 65.56,
    "Delhi": 55.76,
    "Hyderabad": 56.73,
    "Jaipur": 84.87,
    "Kolkata": 65.43,
    "Mumbai": 46.82,
    "Pune": 54.45
}


class PropertyInput(BaseModel):
    City: str
    Locality_Type: str
    Property_Type: str
    BHK: int
    Bathrooms: int
    Super_Area_SqFt: float
    Carpet_Area_SqFt: float
    Floor_Number: int
    Total_Floors: int
    Age_of_Property: int
    Furnishing_Status: str
    Parking: int
    Lift_Available: int
    Gated_Community: int
    Distance_to_Metro_km: float
    Distance_to_City_Center_km: float

    # Price declared by the property owner
    Declared_Price_Lakhs: float


@app.get("/")
def root():
    return {
        "service": "BT Land Registry Property Price Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict_price(property_data: PropertyInput):

    # Check whether city has a calibrated verification threshold
    city = property_data.City

    if city not in CITY_THRESHOLDS:
        return {
            "status": "unsupported_city",
            "message": (
                f"Verification is currently supported only for: "
                f"{', '.join(CITY_THRESHOLDS.keys())}"
            )
        }

    # Convert request into model input
    model_input = pd.DataFrame([{
        "City": property_data.City,
        "Locality_Type": property_data.Locality_Type,
        "Property_Type": property_data.Property_Type,
        "BHK": property_data.BHK,
        "Bathrooms": property_data.Bathrooms,
        "Super_Area_SqFt": property_data.Super_Area_SqFt,
        "Carpet_Area_SqFt": property_data.Carpet_Area_SqFt,
        "Floor_Number": property_data.Floor_Number,
        "Total_Floors": property_data.Total_Floors,
        "Age_of_Property": property_data.Age_of_Property,
        "Furnishing_Status": property_data.Furnishing_Status,
        "Parking": property_data.Parking,
        "Lift_Available": property_data.Lift_Available,
        "Gated_Community": property_data.Gated_Community,
        "Distance_to_Metro_km": property_data.Distance_to_Metro_km,
        "Distance_to_City_Center_km": property_data.Distance_to_City_Center_km
    }])

    # ML prediction
    predicted_price = float(model.predict(model_input)[0])

    declared_price = float(property_data.Declared_Price_Lakhs)

    # Calculate deviation from the ML-estimated price
    deviation_percentage = (
        abs(declared_price - predicted_price)
        / predicted_price
    ) * 100

    # City-specific model error threshold
    threshold = CITY_THRESHOLDS[city]

    # Verification status
    if deviation_percentage > threshold:
        verification_status = "Review Recommended"
        message = (
            "The declared price differs from the model estimate "
            "beyond the calibrated city-specific error range. "
            "Manual review is recommended."
        )
    else:
        verification_status = "Within Model Error Range"
        message = (
            "The declared price is within the calibrated "
            "city-specific model error range."
        )

    return {
        "city": city,
        "declared_price_lakhs": round(declared_price, 2),
        "predicted_price_lakhs": round(predicted_price, 2),
        "predicted_price_inr": round(predicted_price * 100000),
        "deviation_percentage": round(deviation_percentage, 2),
        "city_threshold_percentage": round(threshold, 2),
        "verification_status": verification_status,
        "message": message
    }
