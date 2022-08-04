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
import streamlit.components.v1 as components
import pickle
import matplotlib.pyplot as plt
import requests

# On récupère notre fichier clients pour obtenir les informations descriptives des clients
file_clients_descr = open("application_test.pkl", "rb") #fichier client avec les noms de colonne
donnees_clients_descr = pickle.load(file_clients_descr)
file_clients.close()

# Set FastAPI endpoints : un pour les prédictions, un autre pour les explications
endpoint = 'http://127.0.0.1:8000/predict'
# endpoint = 'https://shielded-bastion-88611.herokuapp.com/predict' # Specify this path for Heroku deployment

endpoint_lime = 'http://127.0.0.1:8000/lime'
# endpoint_lime = 'https://shielded-bastion-88611.herokuapp.com/lime' # Specify this path for Heroku deployment

endpoint_client = 'http://127.0.0.1:8000/client'
# endpoint_client = 'https://shielded-bastion-88611.herokuapp.com/client' # Specify this path for Heroku deployment

endpoint_client_data = 'http://127.0.0.1:8000/clientdata'
# endpoint_client = 'https://shielded-bastion-88611.herokuapp.com/client' # Specify this path for Heroku deployment


# Mise en page de l'application steamlit
st.set_page_config(
    page_title="Le Dashboard de Fatou",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title('Dashboard pour l\'octroi de crédits bancaires')

with st.sidebar:
    NUM_CLIENT = st.number_input("Numéro du client", min_value=0,
                                 help="Entrez le numéro de client de la base de données clients")

    client_json = {'num_client': NUM_CLIENT}

    client_valide = requests.post(endpoint_client, json=client_json,
                                  timeout=8000)

    if client_valide.json()[0]:
        obtain_pred = st.button('Cliquer ici pour connaitre la décision d\'accorder le prêt ou non')
    else:
        st.warning(
            "Veuillez entrer un numéro de client valide pour obtenir des informations concernant la demande d'octroi de prêt")
        obtain_pred = st.button('Cliquer ici pour connaitre la décision d\'accorder le prêt ou non', disabled=True)
if obtain_pred:
    # Il faudra rajouter un test pour voir si le client existe dans la base de données.
    # ligne test qui permet d'afficher le dataframe en cas de tests unitaires

    with st.spinner('Prediction in Progress. Please Wait...'):
        previsions = requests.post(endpoint, json=client_json, timeout=8000)

    container_prev = st.empty()
    with container_prev.container():
        col1, col2 = st.columns([1, 4], gap="medium")

        with col1:
            if previsions.json()[0] == 0:
                indicateur_pret = 'green'
                message="Bravo, votre demande de crédit peut être acceptée"
            else:
                indicateur_pret = 'red'
                message="Malheureusement, votre demande de crédit ne peut être acceptée"

            fig, ax = plt.subplots()
            ax.set(xlim=(-0.1, 0.1), ylim=(-0.1, 0.1))
            a_circle = plt.Circle((0, 0), 0.1, facecolor=indicateur_pret)
            ax.add_artist(a_circle)
            plt.axis('off')
            plt.grid(b=None)
            st.pyplot(fig)
            st.write(message)

        with col2:
            st.write("Ci-dessous les résultats de la prédiction ainsi que les données client")
            st.write("La probabilité de faillite du client est de ", previsions.json()[1])

            donnees_clients = requests.post(endpoint_client_data, json=client_json,
                                            timeout=8000)

            st.table(donnees_clients.json())



    container_explain = st.empty()
    with container_explain.expander("Cliquez ici pour obtenir des explications concernant cette décision"):
        with st.spinner('Veuillez patienter, nous récupérons des données supplémentaires pour expliquer la décision...'):
            output_lime = requests.post(endpoint_lime, json=client_json, timeout=8000)
        import streamlit.components.v1 as components
        components.html(output_lime.json()[0], height=200)