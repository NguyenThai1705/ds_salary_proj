import flask
from flask import Flask, jsonify, request, render_template
import json
import pickle
import numpy as np 
import pandas as pd
from data_input import categorical_cols, columns_model, ohe

app = Flask(__name__)
def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    For using internal requests
    '''
# =============================================================================
#         # stub input features
#     request_json = request.get_json()
#     x = request_json['input']
#     x_in = np.array(x).reshape(1, -1)
#     
#     # load model
#     model = load_models()
#     prediction = model.predict(x_in)[0]
#     response = json.dumps({'response': prediction})
#     return response, 200
#     
# =============================================================================
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        input_features = [[x for x in request.form.values()]]
        
        newdf = pd.DataFrame(input_features, columns=columns_model)
        # Apply ohe on newdf
        cat_ohe_new = ohe.transform(newdf[categorical_cols])
        #Create a Pandas DataFrame of the hot encoded column
        ohe_df_new = pd.DataFrame(cat_ohe_new, columns = ohe.get_feature_names(input_features = categorical_cols))
        #concat with original data and drop original columns
        df_ohe_new = pd.concat([newdf, ohe_df_new], axis=1).drop(columns = categorical_cols, axis=1)

        x_in = df_ohe_new.values.tolist()
        
        # load model
        model = load_models()
        prediction = model.predict(x_in)
        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text='Employee Salary should be $ {}k/year'.format(output))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    application.run(debug=True)