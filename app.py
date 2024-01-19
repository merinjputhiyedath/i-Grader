# Merin Joseph
# CPSC 597- Final project 
# i-Grader
# CSUF - CS Grad

from flask import Flask,request,render_template,url_for,jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import os
import openai
from transformers import pipeline
import torch
from flask_cors import CORS

openai.organization = "#Replace with your OpenAI organization"
openai.api_key = "#Replace with your OpenAI private key"
openai.Model.list()

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/svr": {"origins": "http://localhost:5000"}})

loaded_vectorizer = pickle.load(open(r'C:\Users\ponny\Desktop\final_proj\SavedModels\vectorizer_with_PP.pickle', 'rb'))
lr_model = pickle.load(open(r'C:\Users\ponny\Desktop\final_proj\SavedModels\LR_with_pp','rb'))  
svr_model = pickle.load(open(r'C:\Users\ponny\Desktop\final_proj\SavedModels\SVR_with_pp', 'rb'))
rf_model = pickle.load(open(r'C:\Users\ponny\Desktop\final_proj\SavedModels\RF_with_PP', 'rb'))
# Load the tokenizer and model back using pickle
loaded_ai_tokenizer = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\webapp\aitokenizer.pkl", 'rb'))
loaded_ai_model = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\webapp\aimodel.pkl", 'rb'))

prep_df = pd.read_csv(r'C:\Users\ponny\Desktop\final_proj\Processed_data1.csv')
prep_df.drop('Unnamed: 0',inplace=True,axis=1)
dummy_features = np.zeros((1, len(prep_df.columns)-5))
# prep_df_features  
prep_df_features = prep_df.iloc[:,5:]
question_text = "It should be a 200-400 worded essay discussing the significance of mental health, including an introduction, content, and summary sections. In the essay, be sure to cite at least one external publication or research paper related to mental health to support your points. Essay should explain how a strong focus on mental health can lead to a healthier and more fulfilling life."

def data_preprocess(essay_text):
    count_vectors = loaded_vectorizer.transform([essay_text])
    vectorized_essay = count_vectors.toarray()
    final_text = vectorized_essay.reshape(1,-1)
    X_pred = np.concatenate((dummy_features, final_text), axis=1) 
    return X_pred

@app.route('/', methods=['GET'])
def app_name():
    return "Hi this is Merin" , 200

def ai_detection(data_test):
    inputs = loaded_ai_tokenizer(data_test, return_tensors="pt")
    outputs = loaded_ai_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    if predicted_class == 1:
        return("The sample text is AI-generated or AI-related.")
    else:
        return("The sample text is human-authored or not AI-related.")


@app.route('/lr', methods=['POST'])
def linear_regression():
    try:
        ip_text=request.get_json("text")["text"]
        final_text = data_preprocess(ip_text)
        y_pred = lr_model.predict(final_text)
        return jsonify({'Score using Linear Regression': y_pred.tolist(), 'AI Detection': ai_detection(ip_text)})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/svr', methods=['POST'])
def svr():
    ip_text=request.get_json("text")["text"]
    final_text = data_preprocess(ip_text)
    y_pred=svr_model.predict(final_text)
    return jsonify({'Score using SVR': y_pred.tolist(), 'AI Detection': ai_detection(ip_text)})

@app.route('/rf', methods=['POST'])
def random_forest():
    ip_text=request.get_json("text")["text"]
    final_text = data_preprocess(ip_text)
    y_pred = rf_model.predict(final_text)
    return jsonify({'Score using Random Forest': y_pred.tolist(), 'AI Detection': ai_detection(ip_text)})

@app.route('/gpt', methods=['POST'])
def gpt():
    final_text = request.get_json("text")["text"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an AI essay grader. Score the below essay out of 10 based on the requirements mentioned in the question and give feedback. Detect if the essay was machine-generated or human-authored. Give zero is essay is below 20 words."
            },
            {
                "role": "user",
                "content": "Question: " + question_text + "\n Essay: " + final_text
            }
        ],
        temperature=0.99,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["AI model"]
    )
    generated_message = response['choices'][0]['message']['content']
    return generated_message

if __name__=='__main__':
    app.run(port=5000,debug=True)