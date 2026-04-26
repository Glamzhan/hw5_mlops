import pandas as pd
import yaml
import pickle
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open('params.yaml') as f:
    params = yaml.safe_load(f)

train = pd.read_csv('data/processed/train.csv')
test = pd.read_csv('data/processed/test.csv')

X_train, y_train = train.drop('target', axis=1), train['target']
X_test, y_test = test.drop('target', axis=1), test['target']

mlflow.set_experiment('iris_experiment')

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_param('n_estimators', params['n_estimators'])
    mlflow.log_param('random_state', params['random_state'])
    mlflow.log_metric('accuracy', acc)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    mlflow.log_artifact('model.pkl')
    print('Accuracy:', acc)
