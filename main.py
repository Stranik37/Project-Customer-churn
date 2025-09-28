import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Portfolio/datasets/Customer-Churn-Records.csv') #Читаем данные из csv файла

pd.set_option('display.max_columns', None) #Для удобства отображаем все столбцы

#Очистим в таблице все строки, где есть хотя бы 1 пустое значение
df.dropna()

#Удалим из таблицы все дубликаты
df = df.drop_duplicates()

#Наша главная задача — определить категорию клиентов с наибольшим уровнем оттока.
# Для этого мы поочерёдно проанализируем несколько факторов в таблице,
# а затем объединим результаты, чтобы сделать обоснованные выводы

#Начнем мы с кредитного рейтинга

#Вычислим средний кредитный рейтинг
mean_creditscore = df['CreditScore'].mean().round()

#Создадим 2 группы по значению кредитного рейтинга
high_creditscore = df['CreditScore'] >= mean_creditscore
low_creditscore = df['CreditScore'] < mean_creditscore

#Найдем общее количество клиентов в группах
total_high = high_creditscore.sum()
total_low = low_creditscore.sum()

#Найдем число ушедших клиентов в группах
churn_high = df.loc[high_creditscore & (df['Exited'] == 1)].shape[0]
churn_low = df.loc[low_creditscore & (df['Exited'] == 1)].shape[0]

# Вычисляем уровень оттока в % для каждой группы
churn_rate_high = churn_high / total_high * 100
churn_rate_low = churn_low / total_low * 100

print(f"Отток среди клиентов с высоким кредитным рейтингом: {churn_rate_high:.2f}%")
print(f"Отток среди клиентов с низким кредитным рейтингом: {churn_rate_low:.2f}%")
print()

# Разница в уровнях оттока
diff = churn_rate_low - churn_rate_high
print(f"Разница в уровне оттока между низким и высоким кредитным рейтингом: {diff:.2f}%")
print()

#Как мы видим разница совсем небольшая - 2.03%


#Далее узнаем есть ли отток по значению возраста

#Посчитаем среднее и медианное значение возраста между ушедшими и оставшимися клиентами
mean_age_exit = df.loc[df['Exited'] == 1, 'Age'].mean().round()
median_age_exit = df.loc[df['Exited'] == 1, 'Age'].median().round()

mean_age_not_exit = df.loc[df['Exited'] == 0, 'Age'].mean().round()
median_age_not_exit = df.loc[df['Exited'] == 0, 'Age'].median().round()
print(f"Средний возраст клиентов, ушедших из банка {mean_age_exit} и медианное значение {median_age_exit}")
print(f"Средний возраст клиентов, оставшихся в банка {mean_age_not_exit} и медианное значение {median_age_not_exit}")
print()

min_age = df['Age'].min()
max_age = df['Age'].max()

#Разделяем клиентов на 3 возрастные группы и посчитаем их количество
young_client = df[df['Age'].between(min_age,25)].Age.count()
middle_client = df[df['Age'].between(25,40)].Age.count()
old_client = df[df['Age'].between(40, max_age)].Age.count()

#Найдем число ушедших клиентов в группах
young_client_exit = df[df['Age'].between(min_age,25) & df['Exited'] == 1].Age.count()
middle_client_exit = df[df['Age'].between(25,40) & df['Exited'] == 1].Age.count()
old_client_exit = df[df['Age'].between(40, max_age) & df['Exited'] == 1].Age.count()

# Вычисляем уровень оттока в % для каждой группы
churn_young = round(young_client_exit / young_client * 100)
churn_middle = round(middle_client_exit / middle_client * 100)
churn_old = round(old_client_exit / old_client * 100)

print(f"Отток клиентов в возрасте до 25 составляет: {churn_young}%")
print(f"Отток клиентов в возрасте от 25 до 40 составляет: {churn_middle}%")
print(f"Отток клиентов в возрасте от 40 и больше составляет: {churn_old}%")
print()
#Далее я хочу проверить влияет ли зарплата и наличие кредитной карты на отток клиентов

#Найдем среднюю зарплату
mean_salary = df['EstimatedSalary'].mean().round()

# Создаем колонку с группами
df['SalaryGroup'] = df['EstimatedSalary'].apply(lambda x: 'low_salary' if x < mean_salary else 'high_salary')

# Группируем и считаем общее и ушедших
summary = df.groupby(['SalaryGroup', 'HasCrCard']).agg(
    total_clients=('Exited', 'count'),
    churned_clients=('Exited', 'sum')
).reset_index()

# Вычисляем процент оттока
summary['churn_rate'] = summary['churned_clients'] / summary['total_clients'] * 100

print(summary)

diff = summary.loc[summary['SalaryGroup'] == 'high_salary', 'churn_rate'].values[0] - \
       summary.loc[summary['SalaryGroup'] == 'low_salary', 'churn_rate'].values[0]

print(f"\nРазница в уровне оттока между низкой и высокой зарплатой: {diff:.2f}%")

#Разница совсем небольшая, значит идем далее

#Давайте теперь проверим отток клиентов по географическому расположению

#Сгруппируем ушедших клиентов по странам
Geography_group = df.groupby(['Geography']).agg(
    total_clients=('Exited', 'count'),
    churned_clients=('Exited', 'sum')
).reset_index()

Geography_group['churn_rate'] = Geography_group['churned_clients'] / Geography_group['total_clients'] * 100
#Можно увидеть, что в основном отток клиентов происходит из Германии

#Давайте теперь сгруппируем все полученные нами значения и посмотрим на результат

#Создадим столбец с группами возраста
df['AgeGroup'] = df['Age'].apply(lambda x:
    'young' if min_age <= x <= 25
    else 'middle' if 25 < x <= 40
    else 'old')
#Создадим столбец со значениями кредитного рейтинга
df['CreditScoreGroup'] = df['CreditScore'].apply(lambda x:
    'high_creditscore' if x >= mean_creditscore
    else 'low_creditscore')

finish_churn = df.groupby(['CreditScoreGroup', 'AgeGroup', 'SalaryGroup', 'HasCrCard', 'Geography']).agg(
    total_clients=('Exited', 'count'),
    churned_clients=('Exited', 'sum')
).reset_index()

#Найдем щначение оттока клиентов
finish_churn['churn_rate'] = finish_churn['churned_clients'] / finish_churn['total_clients'] * 100

#Найдем максимальное значение churn_rate
max_churn_rate = finish_churn['churn_rate'].max()

#Выведем значения с максимальным churn_rate
print(finish_churn[finish_churn['churn_rate'] == max_churn_rate].to_string())

#В итоге мы видим: наибольший отток у клиентов в возрасте от 40 лет из Германии,
# с маленькой кредитной историей и зарплатой
# не имеющих кредитных карт

#Сделаем наглядный график по 3 факторам из таблицы
sns.catplot(
    data=finish_churn,
    x='AgeGroup', y='churn_rate',
    hue='Geography',
    col='HasCrCard',
    kind='bar',
    height=5, aspect=1
)
plt.subplots_adjust(top=0.85)
plt.suptitle('Отток по возрасту, географии и наличию карты')
plt.show()

