import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import streamlit as st
st.write("""
# Spam Detector App
""")
st.sidebar.header('User Input Features')
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
n = st.sidebar.slider('No. of rows in the given CSV file', 1, 50, 2)
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    flag = 0
else:
    Email = st.text_input("Enter The Mail you want to Check", "Type Here")
    flag = 1
load_clf = pickle.load(open('spamdetector.pkl', 'rb'))
if flag == 0:
    prediction = load_clf.predict(input_df['Message'])
    dats = [['Sample Category', 'Sample@mail.com']]
    pdf = pd.DataFrame()
    for i in prediction:
        if prediction[i] == 1:
            pdf = pdf.append({'Category': 'This is a Spam', 'Sender': input_df.Sender[i], 'Mail Text': input_df.Message[i]}, ignore_index=True)
        elif prediction[i] == 0:
            pdf = pdf.append({'Category': 'This is Not a Spam', 'Sender': input_df.Sender[i], 'Mail Text': input_df.Message[i]}, ignore_index=True)
    st.write(pdf)
if flag == 1:
    df = pd.read_csv("spam.csv")
    df['Categories'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
    X_train, X_test, y_train, y_test = train_test_split(df.Message,df.Categories)
    v = CountVectorizer()
    X_train_count = v.fit_transform(X_train.values)
    X_train_count.toarray()
    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
        ])
    clf.fit(X_train, y_train)
    abst = clf.predict([Email])
    if abst == 1:
        st.write("The Given Text is a Spam Text!")
    elif abst == 0:
        st.write("The Given Text is Not a Spam Text")
