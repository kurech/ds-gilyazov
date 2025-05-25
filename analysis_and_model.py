import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], inplace=True)
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        data.rename(columns={
            'Air temperature [K]': 'AirTemperature',
            'Process temperature [K]': 'ProcessTemperature',
            'Rotational speed [rpm]': 'RotationalSpeed',
            'Torque [Nm]': 'Torque',
            'Tool wear [min]': 'ToolWear'
        }, inplace=True)

        numerical = ['AirTemperature', 'ProcessTemperature', 'RotationalSpeed', 'Torque', 'ToolWear']
        scaler = StandardScaler()
        data[numerical] = scaler.fit_transform(data[numerical])


        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        }

        results = {}

        st.header("Результаты моделей")

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=False)
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)

            results[name] = {
                'model': model, 'accuracy': acc, 'confusion': cm,
                'report': cr, 'roc_auc': auc, 'fpr': fpr, 'tpr': tpr
            }

            st.subheader(name)
            st.write(f"**Accuracy**: {acc:.3f}")
            st.write("**Classification Report:**")
            st.text(cr)
            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        st.subheader("ROC-кривые")
        fig, ax = plt.subplots()
        for name, res in results.items():
            ax.plot(res['fpr'], res['tpr'], label=f"{name} (AUC = {res['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-кривые моделей")
        ax.legend()
        st.pyplot(fig)

        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков:")
            type_input = st.selectbox("Тип продукта (L=0, M=1, H=2)", [0, 1, 2])
            air_temp = st.number_input("Air temperature [K]")
            proc_temp = st.number_input("Process temperature [K]")
            rot_speed = st.number_input("Rotational speed [rpm]")
            torque = st.number_input("Torque [Nm]")
            tool_wear = st.number_input("Tool wear [min]")

            submit = st.form_submit_button("Предсказать")

            if submit:
                input_df = pd.DataFrame([[type_input, air_temp, proc_temp, rot_speed, torque, tool_wear]],
                                        columns=['Type'] + numerical)
                input_df[numerical] = scaler.transform(input_df[numerical])
                best_model = results['Random Forest']['model']
                pred = best_model.predict(input_df)
                proba = best_model.predict_proba(input_df)[0][1]

                st.write(f"**Предсказание:** {'Отказ (1)' if pred[0] == 1 else 'Нет отказа (0)'}")
                st.write(f"**Вероятность отказа:** {proba:.2f}")
