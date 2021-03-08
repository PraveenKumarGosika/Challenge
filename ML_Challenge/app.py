import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from pickle import dump,load
st.title("PREDICTING  OUTPUT")
st.markdown('<style>body{background-color: lightblue;}</style>',unsafe_allow_html=True)


def display_info():
    column_names=['col1','col2']
    df = pd.DataFrame(columns = column_names)
    col1=st.number_input("Enter your col1 values", -1000.0, 1000.0)
    col2=st.number_input("Enter your col2 values", -1000.0, 1000.0)
    dict1 = {'col1': col1, 'col2': col2}
    df = df.append(dict1, ignore_index = True)
    return df



def predict(df):
    x=df
    x_test=np.array(x)
    classifier=load(open('pickle/dump.pkl','rb'))
    predictions = classifier.predict(np.array(x_test))
    return predictions

def main():
    dataframe=display_info()
    click = st.button('SUBMIT')
    if click:
        Predictions=predict(dataframe)
        if Predictions==0:
            st.write('zero')
        else:
            st.write('one')
if(__name__=='__main__'):
    main()
