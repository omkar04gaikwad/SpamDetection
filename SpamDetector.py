import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv("spam.csv")
df['Categories'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.Categories)
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
email = ['Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!']
emails = pd.read_csv("spam_example.csv")
#email = input("Enter the mail you want to check: \n")
output = clf.predict(emails['Message'])
for i in output:
    if output[i] == 1:
        print("Yes The mail is Spam!")
        print("The Sender is: \n" + str(emails.Sender[i]))
    elif output[i] == 0:
        print("No the mail is not Spam!")
pickle.dump(clf, open('spamdetector.pkl', 'wb'))
