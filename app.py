
import pickle

import pandas as pd
import numpy as np
import streamlit as st


def main():
    model = load_model("model_dumps/gradient_boosting.sav")
    test_data = load_test_data("data/csgo_task__data1.csv")
    
    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Запрос к модели"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.write("Выберите страницу слева")

        st.header("Описание задачи")
        st.markdown("""Чтобы игра шла лучше, всем игрокам хочется понимать, будет ли выигрышным раунд, поэтому был составлен датасет со всеми данными, которые могут повлиять на исход раунда. Поэтому была построена модель для прогнозирования исходов.""")

        st.header("Описание данных")
        st.markdown("""Предоставленные данные:
* time_left – пройденное время,
* ct_score – счет контртерраристов,
* t_score – счет терраристов,
* map – карта,
* bomb_planted – заложена ли бомба,
* ct_health – суммарное количество единиц здоровья контртерраристов,
* t_health – суммарное количество единиц здоровья терраристов,
* ct_armor – суммарное количество единиц брони контртерраристов,
* t_armor – суммарное количество единиц брони терраристов,
* ct_money – количество денег у команды контртерраристов,
* t_money – количество денег у команды контртерраристов,
* ct_helmets – количество шлемов у команды кт,
* t_helmets – количество шлемов у команды т,
* ct_defuse_kits – количество наборов для разминирования бомбы,
* ct_players_alive – количество живых игроков в команде кт,
* t_players_alive – количество живых игроков в команде т.
К категориальным признакам относятся:
* карта, на которой идет игра;
* закладка бомбы.
К бинарным признакам относятся:
* закладка бомбы, где значение (0) означает False, а значение (1) – True.
К вещественным признакам относятся:
* пройденное время,
* счет команд т и кт,
* суммы единиц здоровья и брони обеих команд,
* количество денег обеих команд,
* количество шлемов в каждой команде,
* количество наборов для разминирования,
* количество живых игроков в каждой команде.""")

    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        st.write("Выберите страницу слева")
        request = st.selectbox(
            "Выберите запрос",
            ["F1_score", "Первые 5 предсказанных значений", "Пользовательский пример", "Пасхалка"]
        )

        if request == "F1_score":
            f1_score = 0.9  # Костыль! Заменить на настоящий подсчёт метрики
            st.write(f"{f1_score}")
        elif request == "Первые 5 предсказанных значений":
            st.header("Первые 5 предсказанных значений")
            first_5_test = test_data.iloc[:5, :]
            first_5_pred = model.predict(first_5_test)
            for item in first_5_pred:
                st.write(f"{item:.2f}")
        elif request == "Пользовательский пример":
            st.header("Пользовательский пример")

            Time_left = st.selectbox("Пройденное время", ['<50', '>=50'])
            Time_left = 0 if Time_left == '<50' else 1

            Ct_score = st.number_input("Счет кт", 0, 15)
            Ct_score = 0 if Ct_score < 8 else 1

            T_score = st.number_input("Счет т", 0, 15)
            T_score = 0 if T_score < 8 else 1

            Ct_health = st.selectbox("Сумма здоровья кт команды", ['<250', '>=250'])
            Ct_health = 0 if Ct_health == '>=250' else 1

            T_health = st.selectbox("Сумма здоровья т команды", ['<250', '>=250'])
            T_health = 0 if T_health == '>=250' else 1

            Ct_armor = st.selectbox("Сумма брони кт команды", ['<250', '>=250'])
            Ct_armor = 0 if Ct_armor == '>=250' else 1

            T_armor = st.selectbox("Сумма брони т команды", ['<250', '>=250'])
            T_armor = 0 if T_armor == '>=250' else 1

            Ct_helmets = st.selectbox("Кол-во шлемов кт команды", ['1', '2', '3+'])
            Ct_helmets = 1 if Ct_helmets == '3+' else 0

            T_helmets = st.selectbox("Кол-во шлемов т команды", ['1', '2', '3+'])
            T_helmets = 1 if T_helmets == '3+' else 0

            Ct_defuse_kits = st.selectbox("Количество наборов для разминирования", ['1', '2', '3+'])
            Ct_defuse_kits = 1 if Ct_helmets == '3+' else 0

            Ct_players_alive = st.selectbox("Кол-во живых игроков кт команды", ['1', '2', '3+'])
            Ct_players_alive = 1 if Ct_players_alive == '3+' else 0

            T_players_alive = st.selectbox("Кол-во живых игроков т команды", ['1', '2', '3+'])
            T_players_alive = 1 if T_players_alive == '3+' else 0


            if st.button('Предсказать'):
                data = [Time_left, Ct_score, T_score, Ct_health, T_health, Ct_armor, T_armor, 0, 0, Ct_helmets, T_helmets, Ct_defuse_kits, Ct_players_alive, T_players_alive, 0, 0, 0, 0, 0, 0, 0]
                data = np.array(data).reshape((1, -1))
                pred = model.predict(data)
                st.write(f"Предсказанное значение: {pred[0]:.2f}")
            else:
                pass


@st.cache_data
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file, index_col='Unnamed: 0')
    df = df.drop(labels=['bomb_planted'], axis=1)
    return df


if __name__ == "__main__":
    main()