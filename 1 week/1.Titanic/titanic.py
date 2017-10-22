import pandas
import numpy
import re

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
list_answers = [i for i in range(6)]

list_answers[0] = ([data['Sex'].value_counts()['male'],
                    data['Sex'].value_counts()['female']])

surv_per = (data['Survived'].value_counts()[1] / data.index[-1])*100
list_answers[1] = float(format(round(surv_per, 2), '0.2f'))

first_per = (data['Pclass'].value_counts()[1] / data.index[-1])*100
list_answers[2] = float(format(round(first_per, 2), '0.2f'))

ages = [
        float(format(round(data['Age'].mean(), 2), '0.2f')),
        float(format(round(data['Age'].median(), 2), '0.2f'))
        ]
list_answers[3] = ages

Pcorr = data[['SibSp', 'Parch']].corr()['SibSp'][1]
list_answers[4] = float(format(round(Pcorr, 2), '0.2f'))

FNames = pandas.Series()
dataNames = data[['Name', 'Sex']]
for i in range(data.index[-1]):
    if dataNames.ix[i+1]['Sex'] == 'male':
        pass
    else:
        FNames.set_value(i+1, re.findall("[\w']+", dataNames.ix[i+1]['Name']))

AllFNames = pandas.Series()
for i in range(len(FNames)):
    for j in range(len(FNames.values[i])):
        AllFNames.set_value(i, FNames.values[i][j])

print(AllFNames.value_counts())

list_answers[5] = 'Mary'

for i in range(len(list_answers)):
    with open('ans{0}.txt'.format(i), 'w') as file:
        file.write(str(list_answers[i]))

