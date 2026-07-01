import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

PATH = 'Flood_Prediction_Dataset.csv'

USE_XGBOOST = True
try:
    from xgboost import XGBClassifier
except Exception as exc:
    USE_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier
    print('XGBoost unavailable; falling back to RandomForestClassifier.')
    print('Error details:', exc)

if __name__ == '__main__':
    df = pd.read_csv(PATH)
    X = df.drop('flood', axis=1)
    y = df['flood']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if USE_XGBOOST:
        model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        print('Training XGBoost model...')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print('Training RandomForest model...')

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print('Model accuracy:', round(accuracy_score(y_test, y_pred) * 100, 2), '%')
    print('\nClassification report:\n', classification_report(y_test, y_pred))

    joblib.dump(model, 'floods.save')
    joblib.dump(scaler, 'transform.save')
    print('Saved floods.save and transform.save')
