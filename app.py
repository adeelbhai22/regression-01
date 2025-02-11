from flask import Flask, jsonify,render_template,request
import pickle
import numpy as np
app = Flask('__name__')
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    feature=[int(x) for x in request.form.values()]
    feature_final=np.array(feature).reshape(-1,1)
    prediction=model.predict(feature_final)
    #return render_template('index.html',prediction_text='Price of House will be Rs. {}'.format(int(prediction)))
    return jsonify({"predicted_price": int(prediction[0])})  # Extract first value from NumPy array

if(__name__=='__main__'):
    app.run(debug=True)