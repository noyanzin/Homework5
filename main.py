# Домащнее задание
# 1. Для нашего пайплайна (Case1) поэкспериментировать с разными моделями: 1 - бустинг, 2 - логистическая регрессия (не забудьте здесь добавить в cont_transformer стандартизацию - нормирование вещественных признаков)
# 2. Отобрать лучшую модель по метрикам (кстати, какая по вашему мнению здесь наиболее подходящая DS-метрика)
# 3. Для отобранной модели (на отложенной выборке) сделать оценку экономической эффективности при тех же вводных, как в вопросе 2 (1 доллар на привлечение, 2 доллара - с каждого правильно классифицированного (True Positive) удержанного). (подсказка) нужно посчитать FP/TP/FN/TN для выбранного оптимального порога вероятности и посчитать выручку и траты.
# 4. (опционально) Провести подбор гиперпараметров лучшей модели по итогам 2-3
# 5. (опционально) Еще раз провести оценку экономической эффективности

import xgboost
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from Estimators import *
import itertools
from Processor import *
from sklearn.metrics import \
    f1_score, \
    roc_auc_score, \
    precision_score, \
    classification_report, \
    precision_recall_curve, \
    confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from functions import *
plt.style.use('_mpl-gallery')


categorical_columns = ['Geography', 'Gender', 'Tenure', 'HasCrCard', 'IsActiveMember']
continuous_columns = ['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary']
df = pd.read_csv("churn_data.csv")
# print(df.head(3).T)

X_train, X_test, y_train, y_test = model_selection.train_test_split(df.drop(['Exited'], axis=1), df['Exited'], random_state=0)

models: dict = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42),
    "XGBClassifier": xgboost.XGBClassifier(random_state=42)
}

p = float(input("Сколько стоит удержание 1 человека?"))
e = float(input("Какую прибыль принесет 1 удержанный человек?"))
for m in models:
    am = ModelAnalyzer(m, models[m],
                       continuous_columns=continuous_columns,
                       categorical_columns=categorical_columns)
    am.analyze_model(X_train, y_train, X_test, y_test)
    am.report()
    #мы уже нашли ранее "оптимальный" порог, когда максимизировали f_score
    font = {'size' : 15}

    plt.rc('font', **font)

    cnf_matrix = confusion_matrix(y_test, am.y_pred > am.best_result.threshold)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cnf_matrix, classes=['NonChurn', 'Churn'],
                          title='Confusion matrix')
    plt.savefig("conf_matrix.png")
    plt.show()
    earning = earning_calculate(cnf_matrix, p, e)
    print(f'Для модели {m} доход составит {earning} при стоимости удержания 1 человека{p}, если он принесет доход{e}')



