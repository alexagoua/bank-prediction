from os import write
import streamlit as st
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split


st.write('''
# Application pour prédire les abonnements des clients 
Cette application prédit si un client restera un abonné ou se désabonnera de la banque
''')

st.sidebar.header("Les paramètres d'entrée")

def user_input():
    Total_Relationship_Count=st.sidebar.number_input('Nombre total de produits détenus par le client',1,6000,3000)
    Total_Revolving_Bal=st.sidebar.number_input('Solde renouvelable total sur la carte de crédit',0,2517,1258)
    Total_Amt_Chng_Q4_Q1=st.sidebar.number_input('Changement du montant de la transaction (T4 par rapport au T1)',0,3397,1698)
    Total_Trans_Amt=st.sidebar.number_input('Montant total de la transaction (12 derniers mois)',569,17995,8713)
    Total_Trans_Ct=st.sidebar.number_input('Nombre total de transactions (12 derniers mois)',10,134,62)
    Total_Ct_Chng_Q4_Q1=st.sidebar.number_input('Changement du nombre de transactions (T4 par rapport au T1)',0.0,3.714,1.857)
    Avg_Utilization_Ratio=st.sidebar.number_input("Taux d'utilisation moyen de la carte",0.0,0.999,0.5)
    data={
        'Total_Relationship_Count':Total_Relationship_Count,
        'Total_Revolving_Bal':Total_Revolving_Bal,
        'Total_Amt_Chng_Q4_Q1':Total_Amt_Chng_Q4_Q1,
        'Total_Trans_Amt':Total_Trans_Amt,
        'Total_Trans_Ct':Total_Trans_Ct,
        'Total_Ct_Chng_Q4_Q1':Total_Ct_Chng_Q4_Q1,
        'Avg_Utilization_Ratio':Avg_Utilization_Ratio,
        
    }
    bank_parameters = pd.DataFrame(data,index=[0])
    return bank_parameters

df=user_input()

st.subheader('Résumé des paramètres ajustés')
st.write(df)
valeurs_na = ['Unknown']
df_all = pd.read_csv('Dataset.csv', sep=';',na_values=valeurs_na)
df_all.dropna(inplace=True)
y = df_all['Attrition_Flag']
X= df_all.filter(['Total_Relationship_Count','Total_Revolving_Bal','Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt','Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1)

random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train,y_train)
prediction = random_forest.predict(df)

st.subheader("Le client est:")
if prediction=='Existing Customer':
    st.write("Abonné")
else:
  st.write("Désabonné")
