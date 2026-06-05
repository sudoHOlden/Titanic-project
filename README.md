# 🚢 Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

## 📌 Описание

Проект для **прогнозирования выживаемости пассажиров Титаника** с использованием машинного обучения. Используется классификационная модель Random Forest для анализа того, какие пассажиры выжили при кораблекрушении на основе их характеристик.

## 🎯 Цель

Построить классификационную модель, которая на основе характеристик пассажира (возраст, пол, класс каюты, цена билета, портны посадки) сможет предсказать, выжил ли он при катастрофе.

## 📊 Датасет

**Источник:** Titanic-Dataset.csv

**Исходные признаки:**
- `PassengerId` — ID пассажира
- `Sex` — пол
- `Age` — возраст
- `SibSp` — количество братьев/сестер/супругов
- `Parch` — количество родителей/детей
- `Fare` — цена билета
- `Embarked` — порт посадки (S, C, Q)
- `Cabin` — номер каюты *(удалена)*
- `Ticket` — номер билета *(удалена)*
- `Name` — имя *(удалена)*

**Целевая переменная:** `Survived` (0 — не выжил, 1 — выжил)

## 🔧 Обработка данных

```python
✅ Удалены ненужные столбцы: Cabin, PassengerId, Ticket, Name
✅ Заполнены пропуски в Age медианой
✅ Заполнены пропуски в Embarked значением 'S' (самый частый порт)
✅ Преобразовано Sex: female=1, male=0
✅ One-hot encoding для Embarked (создана Embarked_C, Embarked_Q, Embarked_S)