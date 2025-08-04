from flask import Flask, render_template, request, redirect, url_for
import joblib
import logging
import sys
import os
if sys.stderr is None:
    sys.stderr = open(os.devnull,'w')
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recog', methods=['POST'])
def recog():
    try:
        clf = joblib.load('decision_tree_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        label_mapping = joblib.load('label_mapping.pkl')
        new_text = request.form['text']
        #print(new_text)
        #new_texts=[new_text]  
        new_texts_vect = vectorizer.transform(['new_text'])
        predictions = clf.predict(new_texts_vect)
        predicted_emotions = [label_mapping[pred] for pred in predictions]
        print(f"Predictions: {predictions}")
        result=predicted_emotions[0]
    
    except Exception as e:
        print(f"Error: {e}")
        return str(e)  
    
    return redirect(url_for('result', result=result))
@app.route('/result')
def result():
    result = request.args.get('result')
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
 
