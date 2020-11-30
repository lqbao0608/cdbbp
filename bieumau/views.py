from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index(request):
    context = {}
    return render(request, 'bieumau.html', context)

def chandoan(request):
    context = {}
    convert_data = {
        'FCVC': {
            'Never': 0,
            'Sometimes': 1,
            'Always': 2
        },
        'NCP': {
            '1 to 2': 0,
            '2 to 3': 1,
            'more 3': 2
        },
        'CH2O': {
            'less 1': 0,
            '1 to 2': 1,
            'more 2': 2
        },
        'FAF': {
            '0': 0,
            '1 to 2': 1,
            '2 to 4': 2,
            '4 to 5': 3
        },
        'TUE': {
            '0 to 2': 0,
            '3 to 5': 1,
            'more 5': 2
        }
    }
    if request.method == 'POST':
        req = request.POST.dict()
        # print(req)
        le_arr = get_label_encoder(load_data())
        p = predict_input(
            load_data(),
            [
                le_arr['Gender'].fit_transform([req['Gender']])[0],
                req['Age'],
                req['Height'],
                req['Weight'],
                le_arr['family_history_with_overweight'].fit_transform([req['family_history_with_overweight']])[0],
                le_arr['FAVC'].fit_transform([req['FAVC']])[0],
                convert_data['FCVC'][req['FCVC']],
                convert_data['NCP'][req['NCP']],
                le_arr['CAEC'].fit_transform([req['CAEC']])[0],
                le_arr['SMOKE'].fit_transform([req['SMOKE']])[0],
                convert_data['CH2O'][req['CH2O']],
                le_arr['SCC'].fit_transform([req['SCC']])[0],
                convert_data['FAF'][req['FAF']],
                convert_data['TUE'][req['TUE']],
                le_arr['CALC'].fit_transform([req['CALC']])[0],
                le_arr['MTRANS'].fit_transform([req['MTRANS']])[0]
            ]
        )
        context = {
            'predict': le_arr['NObeyesdad'].inverse_transform(p)[0]
        }
    return render(request, 'bieumau.html', context=context)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
# import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# !pip install imbalanced-learn
# import imblearn

from collections import Counter

# from imblearn.over_sampling import SMOTE



def load_data():
    return pd.read_csv("./bieumau/ObesityDataSet_raw_and_data_sinthetic.csv")

def get_label_encoder(dt):
    le_arr = {}
    for i in dt.columns:
        le_arr[i] = LabelEncoder()
    dt_ft = dt.copy()
    dt_ft['Gender'] = le_arr['Gender'].fit_transform(dt['Gender'])
    dt_ft['family_history_with_overweight'] = le_arr['family_history_with_overweight'].fit_transform(
        dt['family_history_with_overweight'])
    dt_ft['FAVC'] = le_arr['FAVC'].fit_transform(dt['FAVC'])
    dt_ft['CAEC'] = le_arr['CAEC'].fit_transform(dt['CAEC'])
    dt_ft['SMOKE'] = le_arr['SMOKE'].fit_transform(dt['SMOKE'])
    dt_ft['SCC'] = le_arr['SCC'].fit_transform(dt['SCC'])
    dt_ft['CALC'] = le_arr['CALC'].fit_transform(dt['CALC'])
    dt_ft['MTRANS'] = le_arr['MTRANS'].fit_transform(dt['MTRANS'])
    dt_ft['NObeyesdad'] = le_arr['NObeyesdad'].fit_transform(dt['NObeyesdad'])
    return le_arr

def predict_input(dt, data_input):
    le_arr = get_label_encoder(load_data())

    dt_ft = dt.copy()
    dt_ft['Gender'] = le_arr['Gender'].fit_transform(dt['Gender'])
    dt_ft['family_history_with_overweight'] = le_arr['family_history_with_overweight'].fit_transform(
        dt['family_history_with_overweight'])
    dt_ft['FAVC'] = le_arr['FAVC'].fit_transform(dt['FAVC'])
    dt_ft['CAEC'] = le_arr['CAEC'].fit_transform(dt['CAEC'])
    dt_ft['SMOKE'] = le_arr['SMOKE'].fit_transform(dt['SMOKE'])
    dt_ft['SCC'] = le_arr['SCC'].fit_transform(dt['SCC'])
    dt_ft['CALC'] = le_arr['CALC'].fit_transform(dt['CALC'])
    dt_ft['MTRANS'] = le_arr['MTRANS'].fit_transform(dt['MTRANS'])
    dt_ft['NObeyesdad'] = le_arr['NObeyesdad'].fit_transform(dt['NObeyesdad'])

    dt2 = dt_ft.copy()

    # an rau
    dt2.loc[dt2['FCVC'] < 1, ['FCVC']] = 0
    dt2.loc[(dt2['FCVC'] >= 1) & (dt2['FCVC'] < 2), ['FCVC']] = int(1)
    dt2.loc[dt2['FCVC'] >= 2, ['FCVC']] = 2

    # bua an chinh
    dt2.loc[(dt2['NCP'] >= 1) & (dt2['NCP'] < 2), ['NCP']] = 0
    dt2.loc[(dt2['NCP'] >= 2) & (dt2['NCP'] < 3), ['NCP']] = 1
    dt2.loc[dt2['NCP'] >= 3, ['NCP']] = 2

    # uong nuoc
    dt2.loc[dt2['CH2O'] < 1, ['CH2O']] = 0
    dt2.loc[(dt2['CH2O'] >= 1) & (dt2['CH2O'] < 2), ['CH2O']] = 1
    dt2.loc[dt2['CH2O'] >= 2, ['CH2O']] = 2

    # tap the duc
    dt2.loc[dt2['FAF'] < 1, ['FAF']] = 0
    dt2.loc[(dt2['FAF'] >= 1) & (dt2['FAF'] < 2), ['FAF']] = 1
    dt2.loc[(dt2['FAF'] >= 2) & (dt2['FAF'] < 4), ['FAF']] = 2
    dt2.loc[dt2['FAF'] >= 4, ['FAF']] = 3

    # su dung thiet bi cong nghe
    dt2.loc[(dt2['TUE'] >= 0) & (dt2['TUE'] < 2), ['TUE']] = 0
    dt2.loc[(dt2['TUE'] >= 2) & (dt2['TUE'] < 4), ['TUE']] = 1
    dt2.loc[(dt2['TUE'] >= 4), ['TUE']] = 2

    dt2 = dt2.astype({
        'FCVC': 'int32',
        'NCP': 'int32',
        'CH2O': 'int32',
        'FAF': 'int32',
        'TUE': 'int32'
    })

    X = dt2.drop(['NObeyesdad'], axis=1)
    y = dt2['NObeyesdad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf.fit(X_train, y_train)

    return clf.predict([data_input])