from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
model = xgb.Booster(model_file='xgb_model.bin')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    # Get the uploaded CSV file
    file = request.files['file']

    # Read the CSV file into a DataFrame
    test_fault_x = pd.read_csv(file)

    # Make predictions using the trained model
    y_pred = model.predict(xgb.DMatrix(test_fault_x))
    yprob = np.argmax(y_pred, axis=1)
    predictions = [round(value) for value in yprob]

    # Calculate performance metrics
    test_fault_y = pd.read_csv('fault7_output.csv')
    acc, recall, f1, precision, confusion_mat = show_performance(test_fault_y, predictions)

    # Prepare the response
    response = {
        'predictions': predictions,
        'accuracy': acc,
        'recall': recall,
        'f1_score': f1,
        'precision': precision,
        'confusion_matrix': confusion_mat.tolist()
    }

    return jsonify(response)

def show_performance(label, preds):
    accuracy = accuracy_score(label, preds)

    acc = accuracy * 100.0
    recall = metrics.recall_score(label, preds, average='macro')
    f1 = metrics.f1_score(label, preds, average='macro')
    precision = metrics.precision_score(label, preds, average='macro')
    res_confusion_matrix = confusion_matrix(label, preds)

    return acc, recall, f1, precision, res_confusion_matrix

if __name__ == '__main__':
    app.run(debug=True)
