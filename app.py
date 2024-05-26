from flask import Flask,render_template,request
import pickle
import joblib
import pandas as pd

# Load the trained model and CountVectorizer
rf_classifier = joblib.load('rf_classifier.pkl')
cv = joblib.load('count_vectorizer.pkl')



# model = pickle.load(open('rf_classifier.pkl'))

app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # after .get we use the name argument in html
    review=request.form.get('review')

    # now prediction

    dic={'review':review}
    sample_df = pd.DataFrame([dic])
    sample_test=cv.transform(sample_df['review'])
    sample_test=sample_test.toarray()
    sample_to_be_predicted=sample_test
    prediction = rf_classifier.predict(sample_to_be_predicted)
    if prediction==0:
        prediction =  'Negative'
    else:
        prediction =  'Positive'
    
    # to_predict=put_sample_here_to_get_df(s)
    return render_template('index.html',result=prediction)
