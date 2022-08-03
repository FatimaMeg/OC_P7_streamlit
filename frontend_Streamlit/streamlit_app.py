# =========================================
# Dashboard pour l'octroi de crédits bancaires
# Author: Fatima Meguellati
# Last Modified: 03 Aout 2022
# =========================================
# Command to execute script locally: streamlit run app.py
# Command to run Docker image: docker run -d -p 8501:8501 <streamlit-app-name>:latest


import joblib
import streamlit as st
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import pickle
import matplotlib.pyplot as plt
import requests

st.title('Dashboard pour l\'octroi de crédits bancaires')

# Set FastAPI endpoint
#endpoint = 'http://localhost:8000/predict'
endpoint = 'http://host.docker.internal:8000/predict' # Specify this path for Dockerization to work

# On charge notre modèle de prévision
model_pipeline = joblib.load('pipeline_bank_lgbm.joblib')

# On a besoin du fichier de données 'train' pour obtenir les explanations de lime, récupéré au format pickle
file_X_train = open("X_train_Nono2.pkl", "rb")
donnees_train = pickle.load(file_X_train)
file_X_train.close()

# On importe les features importantes récupérées du modèle dans un format pickle
file_features = open("features.pkl", "rb")
features = pickle.load(file_features)
file_features.close()

# On récupère notre fichier clients pour obtenir les informations descriptives des clients
file_clients = open("fichierClient.pkl", "rb")
donnees_clients = pickle.load(file_clients)
file_clients.close()

with st.sidebar:
	NUM_CLIENT = st.number_input("Renseigner ci-dessous le numéro de client pour obtenir"
								 " le dashboard associé à ce client")

	obtain_pred = st.button('Cliquer ici pour connaitre la décision d\'accorder le prêt ou non')

if NUM_CLIENT !='':
    #Il faudra rajouter un test pour voir si le client existe dans la base de données.
	#ligne test qui permet d'afficher le dataframe en cas de tests unitaires
	
	client_json = {'num_client': NUM_CLIENT}
	with st.spinner('Prediction in Progress. Please Wait...'):
		output = requests.post(endpoint, json=client_json,
						   timeout=8000)
		#result2 = num_client_scoring(model, NUM_CLIENT)
		#st.write(result2)

	st.write(output.json())


	#Bouton permettant de générer les explanations du model
	explain_pred = st.button('Cliquer ici pour obtenir des explications')

	#Si l'utilisateur appuie sur le bouton explain predictions, on lui affiche les explications
	if explain_pred:
		with st.spinner('Generating explanations'): #permet d'informer l'utilisateur que le calcul prend un peu de temps
			data = donnees_clients.loc[donnees_clients['SK_ID_CURR'] == 100028, features]
			explainer = lime_tabular.LimeTabularExplainer(donnees_train,mode="classification",class_names=features)
			exp = explainer.explain_instance(data.values[0],
				model_pipeline.predict_proba, num_features=20)
			mongraph_html = exp.as_html()
			components.html(mongraph_html, height=100)

			st.pyplot(exp.as_pyplot_figure())
