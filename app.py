from io import BytesIO

import pandas as pd
from flask import Flask, render_template, request, jsonify, make_response
import json

from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import numpy as np
import shap
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import base64

from package.models import load_model
from package.feature_extraction import get_features
from package.models import read_data

data = read_data()
print(data.shape)
# features = get_features(data)
list_id_client = list(data['SK_ID_CURR'].unique())
features = data.drop(['TARGET', 'SK_ID_CURR'], axis=1).columns.tolist()
print(len(features))
# data = data[features].copy()
X = data[features].values
X_scaled = StandardScaler().fit_transform(X)
print(X.shape)
print(X_scaled.shape)
model = load_model()

# flask
app = Flask(__name__)

seuil = 0.52


@app.route('/')
def home():
    # return render_template('index.html')
    return ("Application flask")


@app.route('/predict/', methods=['GET'])
def predict():
    """
    For rendering results on HTML GUI
    """
    id = request.args['id_client']
    id = int(id)
    if id not in list_id_client:
        # prediction = "Ce client n'est pas répertorié"
        score_dct = {'score': -1}
        return jsonify(json.loads(json.dumps(score_dct)))
    else:
        client_index = data[data['SK_ID_CURR'] == id].index
        # client = client.drop(['SK_ID_CURR'], axis=1)
        # probability_default_payment = model.predict_proba(client)[:, 1]
        probability_default_payment = model.predict_proba(X_scaled[client_index])[:, 1]

        if probability_default_payment >= seuil:
            # prediction = "Client défaillant: prêt refusé"
            score_dct = {'score': 1}
            return jsonify(json.loads(json.dumps(score_dct)))
        else:
            # prediction = "Client non défaillant:  prêt accordé"
            score_dct = {'score': 0}
            # return render_template('index.html', prediction_html=prediction)
            return jsonify(json.loads(json.dumps(score_dct)))


def df_global_importance(n):
    coef = list(model.named_steps['classifier'].coef_[0])
    df_global = pd.DataFrame(zip(features, coef), columns=["feature", "importance"])
    df_global = df_global.sort_values("importance", ascending=False, key=abs)
    return df_global.head(n)


# http://127.0.0.1:5000/id_data/?id=100139
@app.route('/id_data/')
def id_data():
    id_client = int(request.args.get('id'))
    features = data[data['SK_ID_CURR'] == id_client].drop(columns=['TARGET'])
    # status = data.loc[data['SK_ID_CURR'] == id_client, "TARGET"]
    features_json = json.loads(features.to_json())
    # status_json = json.loads(status.to_json())
    return jsonify({"data": features_json})


@app.route('/shap_local/', methods=['GET'])
def plot_shap_local():
    id = int(request.args['id_client'])
    features_idx = data[data['SK_ID_CURR'] == id].drop(['TARGET', 'SK_ID_CURR'], axis=1)
    explainer = shap.Explainer(model.named_steps['classifier'], X, feature_names=features)
    shap_values_idx = explainer(features_idx)
    shap.summary_plot(shap_values_idx, X, plot_type="bar", show=False)
    buf = BytesIO()
    plt.savefig(buf,
                format="png",
                dpi=150,
                bbox_inches='tight')
    dataToTake = base64.b64encode(buf.getbuffer()).decode("ascii")
    return (f"<img src='data:image/png;base64,{dataToTake}'/>")


# http://127.0.0.1:5000/lime_local/?id=100139&n=5
@app.route('/lime_local/', methods=['GET'])
def plot_lime_local():
    id = int(request.args.get('id'))
    n = int(request.args.get('n'))
    explainer = LimeTabularExplainer(X_scaled,
                                     mode='classification',
                                     class_names=['Accordé', 'Refusé'],
                                     feature_names=features)
    idx = data[data["SK_ID_CURR"] == id].index
    df_instance = X_scaled[idx].reshape(len(features), )
    explanation = explainer.explain_instance(df_instance, model.predict_proba, num_features=n)
    explanation_html = explanation.as_html()
    response = make_response(explanation_html)
    return response


# https://127.0.0.1:5000/features/?n=5
@app.route('/features/')
def features_global():
    n = int(request.args.get('n'))
    df_global = df_global_importance(n)
    features = df_global["feature"].tolist()
    dict = {"n": features}
    return jsonify(json.loads(json.dumps(dict)))


# http://127.0.0.1:5000/global/?n=5
@app.route('/global/')
def plot_global():
    n = int(request.args.get('n'))
    df_global = df_global_importance(n).sort_values("importance", ascending=False)

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    sns.barplot(data=df_global, x='importance', y='feature')
    plt.xticks(rotation=25)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    info = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{info}'/>"


# http://127.0.0.1:5000/ids/
@app.route('/ids/')
def list_ids():
    ids_list = data['SK_ID_CURR'].sort_values()
    ids_list_client_json = json.loads(ids_list.to_json())
    return jsonify(ids_list_client_json)


@app.route('/dists/')
def plot_dist():
    n = int(request.args.get('n'))
    id = int(request.args.get('id'))

    g_importance = df_global_importance(n)
    features = g_importance["feature"].tolist()

    idx = data[data["SK_ID_CURR"] == id].index

    fig = plt.figure(figsize=(14, 20))
    try:
        for i, col in enumerate(features):
            plt.subplot(10, 3, i + 1)
            p = sns.histplot(data[col], label=col, color='blue', bins='auto')
            patch_index = data.iloc[idx][col].to_numpy().item()
            print(patch_index)
            for rectangle in p.patches:
                if rectangle.get_x() == patch_index:
                    rectangle.set_color('r')
            plt.legend()
            plt.title(col)
            plt.tight_layout()
    except Exception as e:
        print(col, e)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    info = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{info}'/>"


# Define endpoint for flask
app.add_url_rule('/predict', 'predict', predict)

# Define endpoint for flask
app.add_url_rule('/predict', 'predict', predict)

# lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
