import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score,confusion_matrix
import nltk
# nltk.download('stopwords')

temp_df = pd.read_csv('imbd_dataset.csv')
df = temp_df.iloc[:10000] #Take only 10k values as model will not train fast
print(df.head())

import re
def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text
df['review']=df.review.apply(remove_tags)

df['review'] = df['review'].apply(lambda x:x.lower())

from nltk.corpus import stopwords
sw_list = stopwords.words('english')

df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))

X = df.iloc[:,0:1] #selects only the first column includes 0 and excludes 1 "iloc[rows:columns]"
y = df['sentiment']

#now encoding the sentiments to 0 and 1 so it is easy for the compyter to understand
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

# This line fits the CountVectorizer cv to the training data X_train['review'] and transforms it into a bag-of-words representation. This means it learns the vocabulary from the training data and then transforms the training text data into a matrix of token counts.
X_train_bow = cv.fit_transform(X_train['review'])
X_test_bow = cv.transform(X_test['review'])
# This line transforms the test data into a bag-of-words representation ie into matrix of 0 and 1. 

X_train_bow = X_train_bow.toarray()
X_test_bow = X_test_bow.toarray()


# Now we use random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_bow,y_train)
y_pred = rf.predict(X_test_bow)


joblib.dump(rf, 'rf_classifier.pkl')
joblib.dump(cv, 'count_vectorizer.pkl')