from flask import Flask,jsonify,request
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('ASD_model','rb'))
app = Flask(__name__)
@app.route('/ASD_detection', methods=['POST'])
def apicall():
    
        data = request.get_json(force=True)
        predict_request = [[data['q1'],data['q2'],data['q3'],data['q4'],data['q5'],data['q6'],data['q7'],data['q8'],data['q9'],data['q10'],data['score'],data['age'],data['gen'],data['bwj'],data['fmpd']]]
        predict_request = np.array(predict_request)
        pred = model.predict(predict_request)
        if pred == '1':
            output = 'Yes'
        else :
           output = 'No'
        return jsonify(results=output)

if __name__=='__main__':
    app.run(debug=True)

