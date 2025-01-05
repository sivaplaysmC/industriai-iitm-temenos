from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import os

app = Flask(_name_)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('uploads', f.filename))
        return redirect(url_for('select_model'))

@app.route('/select_model')
def select_model():
    return render_template('select_model.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    model_name = request.form['model']
    file_path = os.path.join('uploads', os.listdir('uploads')[0])
    
    data = pd.read_csv(file_path)
    target = data['loan_status']  
    features = data[['Principal', 'terms', 'age', 'education']] 
    
    label_encoder = LabelEncoder()
    features['education'] = label_encoder.fit_transform(features['education'])

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    if model_name == 'SVM':
        model = SVC()
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=15)
    elif model_name == 'LogisticRegression':
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = LogisticRegression(solver='liblinear')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred, pos_label='PAIDOFF')
    accuracy = accuracy_score(y_test, y_pred)
    
    return render_template('results.html', f1=f1, accuracy=accuracy)

if _name_ == '_main_':
    app.run(debug=True)
