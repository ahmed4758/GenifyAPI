from flask import Flask, request, jsonify
import numpy as np
import pickle
from modelfun import *
import pandas as pd
import xgboost as xgb

app=Flask(__name__)
model = pickle.load(open('gen.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
        if request.method == 'POST':
                xgtest = xgb.DMatrix(test_X)
                preds = model.xgb.predict(xgtest)
                out=preds  
        return jsonify({'prediction': str(out)})


if __name__=="__main__":
        data_path = "./"
        train_file =  open(data_path + "train_ver2.csv")
        x_vars_list, y_vars_list, cust_dict = processData(train_file, {})
        train_X = np.array(x_vars_list)
        train_y = np.array(y_vars_list)
        del x_vars_list, y_vars_list
        test_file = open(data_path + "test_ver2.csv")
        x_vars_list, y_vars_list, cust_dict = processData(test_file, cust_dict)
        test_X = np.array(x_vars_list)
        del x_vars_list
        test_file.close()
        app.run(debug=True)


    