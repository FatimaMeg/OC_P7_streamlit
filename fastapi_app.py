# FastAPI application

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Model import Features, Client
import joblib
import pickle

# 2. Create app and model objects
app = FastAPI()

# On charge notre modèle de prévision
model_pipeline = joblib.load('pipeline_bank_lgbm.joblib')

# On récupère notre fichier clients pour obtenir les informations descriptives des clients
file_clients = open("fichierClient.pkl", "rb")
donnees_clients = pickle.load(file_clients)
file_clients.close()

# On récupère nos features calculées par le modèle au format pkl
file_features = open("features.pkl", "rb")
features = pickle.load(file_features)
file_features.close()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted classification (yes or not) with the confidence
@app.post('/predict')
def predict_clientscoring_features(client: Client):
    # On récupère les features du client
    data = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == client.num_client, features]
    prediction = model_pipeline.predict(data)[0]
    proba = model_pipeline.predict_proba(data)

    if prediction > 0.5:
        prediction_text = "OUI"
    else:
        prediction_text = "NON"

    return {
        'Le client risque-t-il la faillite' : prediction_text,\
		'Sa probabilite de faillite est de ': f"{proba[0][1]*100:.2f} %"
    }


@app.get("/")
async def main():
    content = """
    <body>
    <h2> Bienvenue sur l'API permettant d'obtenir des prévisions d'octroi de prêt</h2>
    <p> You can view the FastAPI UI by heading to localhost:8000 </p>
    <p> Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests </p>
    </body>
    """
    return ("Bienvenue dans mon API")

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
