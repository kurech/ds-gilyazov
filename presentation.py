import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Задача: бинарная классификация (отказ/не отказ)
    - Цель: предиктивное обслуживание на производстве
    ---
    ## Данные
    - AI4I 2020 Predictive Maintenance Dataset
    - 10 000 записей, 14 признаков
    ---
    ## Этапы работы
    1. Загрузка и предобработка данных
    2. Обучение моделей
    3. Оценка качества
    4. Визуализация и предсказание
    ---
    ## Приложение
    - Streamlit
    - Страницы: анализ и презентация
    ---
    ## Вывод
    - Успешно обучены модели
    - Возможности для улучшения: больше признаков, продвинутая обработка
    """

    with st.sidebar:
        st.header("Настройки")
        theme = st.selectbox("Тема", ["white", "black", "serif", "simple", "sky", "beige"])
        height = st.number_input("Высота", value=500)
        transition = st.selectbox("Переход", ["slide", "zoom", "fade", "convex", "concave", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "zoom", "notes", "search"], [])

    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={"transition": transition, "plugins": plugins},
        markdown_props={"data-separator-vertical": "^--$"},
    )
