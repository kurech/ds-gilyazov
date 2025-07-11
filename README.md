# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## 📌 Описание проекта

Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдёт ли отказ оборудования (**Target = 1**) или нет (**Target = 0**).  
Результаты работы оформлены в виде многостраничного веб-приложения, созданного с использованием библиотеки **Streamlit**.

Приложение позволяет:

- загружать и анализировать данные;
- обучать модели и сравнивать результаты;
- делать предсказания по новым входным данным;
- просматривать краткую презентацию проекта.

---

## 📊 Описание датасета

Используется датасет **AI4I 2020 Predictive Maintenance Dataset**, содержащий 10 000 записей с 14 признаками:

- Тип продукта (`Type`)
- Температура окружающей среды (`Air temperature`)
- Температура процесса (`Process temperature`)
- Скорость вращения (`Rotational speed`)
- Крутящий момент (`Torque`)
- Износ инструмента (`Tool wear`)
- Целевая переменная: `Machine failure` (0 или 1)

📁 Источник: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)

---

## ⚙️ Установка и запуск

1. Клонируйте репозиторий:

```bash
git clone https://github.com/kurech/ds-gilyazov.git
cd ds-gilyazov
```

2. Создайте и активируйте виртуальное окружение (опционально):

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```

4. Запустите приложение:

```bash
streamlit run app.py
```

После запуска откроется веб-интерфейс в браузере по адресу http://localhost:8501.

---

## 🗂 Структура репозитория

```bash
ds-gilyazov/
│
├── app.py                  # Главный файл Streamlit-приложения
├── analysis_and_model.py  # Страница анализа данных, моделей и предсказаний
├── presentation.py        # Презентация проекта (слайды)
├── requirements.txt       # Зависимости
├── README.md              # Этот файл
└── data/
    └── predictive_maintenance.csv  # Датасет (если не загружается вручную)

```

---

## 🧠 Используемые модели

Проект включает обучение и сравнение следующих моделей:

- Логистическая регрессия (Logistic Regression)
- Случайный лес (Random Forest Classifier)
- XGBoost Classifier

Для оценки моделей рассчитываются:

- Accuracy
- Classification Report
- Confusion Matrix
- ROC-AUC и графики ROC-кривых

---

## 📈 Презентация проекта

Во вкладке "Презентация" находится краткая интерактивная демонстрация этапов работы над проектом в формате слайдов (реализовано через streamlit-reveal-slides).

---

## 🧪 Предсказание

Во вкладке "Анализ и модель" доступна форма для ввода новых данных. После заполнения полей отображаются:

- Предсказанный результат: отказ (1) / не отказ (0)
- Вероятность отказа

---

## 🎬 Видео-демонстрация

Просмотреть видео можно на [YouTube](https://youtu.be/fcQ-h6JkY2g?si=wnF6Jq-K31M2dHCF)
