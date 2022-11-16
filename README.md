# Регрессия
Регрессия – это задача машинного обучения, которая заключается в предсказании некоторой целевой переменной, на основе одной или нескольких независимых переменных.  
Задача регрессии относится к категории задач обучения с учителем. Это значит, что набор данных, который используется для обучения, должен иметь определенную структуру, который обычно представляется в виде таблицы. В столбцах таблицы находятся различные характеристики (атрибуты) объектов, а на пересечениях строк и столбцов – значения данной характеристики у данного объекта. Один из этих атрибутов и является целевой переменной, значение и предсказывается с помощью модели машинного обучения.
Линейная регрессия - это метод, используемый для анализа линейной зависимости между входными переменными (независимыми переменными) и одной целевой переменной. Линейная зависимость означает, что точки данных имеют тенденцию следовать прямой линии. Простая линейная регрессия включает только одну входную переменную.
Математическое уравнение, которое оценивает линию простой линейной регрессии выглядит как:
```
y=a+bx
```
`x` - это независимая переменная (предиктор).  
`y` – это зависимая переменная (переменная отклика).  `y` это ожидаемое значение при известном `x`.  
`a` и `b` – это некоторые действительные числа, из которых a представляет собой смещение, а b угловой коэффициент (градиент).  
Если в линейной регрессии содержится более одной независимой переменной, то ей называют множественной линейной регрессией.
В линейной регрессии цель - это найти линию, которая наилучшим образом моделирует путь точек данных. Положение линии определяется определенными коэффициентами подгонки, а линейная регрессия помогает подобрать подходящие значения для этих коэффициентов.
# Пример простой линейной регрессии
Ниже приведён пример реализации модели простой линейной регрессии на языке Python и библиотеки Scikit-Learn.
Для начала нужно импортировать необходимые пакеты и функции.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```
Далее необходимо загрузить набор данных.
```python
dataset = pd.read_csv('D:\\data.csv', encoding = 'ISO-8859-1')
```
С помощью следующего кода можно отобразить данные из набора данных в виде графика.
```python
dataset.plot(x='time_study', y='Marks', style='bo')
plt.show()
```
![Рисунок 1](https://github.com/Erkeri/LabNeuralNetwork/raw/master/Images/image1.png)  
Исходя из графика можно предположить, что между значениями time_study и Marks существует линейная зависимость.
Далее необходимо подготовить данные. В начале необходимо отделить зависимые переменные от независимых.
```python
x = dataset[['time_study']]
y = dataset['Marks']
```
После необходимо разделить данные на тренировочные и тестовые, для этого можно использовать всторенные метод библиотеки Scikit-Learn - train_test_split.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
После разделения данных можно преступить к обучению модели. Нужно создать экземпляр класса Linear Regression и вызвать метод fit передав в него данные для обучения.
```python
model = LinearRegression()
model.fit(x_train, y_train)
```
После обучения уже можно сделать прогноз на тестовых данных. Для этого нужно вызвать метод predict у натренированной модели. Этот метод возвращает предсказанные значения y_pred для всех входных значений x_test.
```python
y_pred = model.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
```
![Рисунок 2](https://github.com/Erkeri/LabNeuralNetwork/raw/master/Images/image2.png)  
Для оценки алгоритма можно использовать следующий код. 
```python
print('Mean absolute error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean squared error:', metrics.mean_squared_error(y_test, y_pred))
print('Root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```
Результат можно видеть на скриншоте ниже.  
![Рисунок 3](https://github.com/Erkeri/LabNeuralNetwork/raw/master/Images/image3.png)  
Для реализации множественной линейной регрессии нужно использовать приведённый выше и набор данных с несколькими независимыми переменными. Так же нужно изменить часть с отделение независимых переменных от зависимой переменной следующим образом.
```python
x = dataset[['Year', 'Selling_Price', 'Driven_kms', 'Fuel_Type_0', 'Selling_type_0', 'Transmission_0']]
y = dataset['Present_Price']
```
Регрессионный анализ — важный статистический метод, широко используемый в статистике и бизнесе. Это обязательный инструмент в нашем наборе инструментов для обработки данных. В данном руководстве описан способ реализация линейной регрессии с помощью библиотеки Scikit-Learn.