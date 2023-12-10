import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import statistics
import time




# Зчитування даних
df = pd.read_csv('Diabets_World (2).csv')
plt.figure()


column_years_data = df['year'].tolist()
column_index_data = df['index'].tolist()

test_ratio = 0.2
train_ratio = 0.8

# Розрахунок кількості рядків для тестової і дослідної вибірок
test_size = int(len(df) * test_ratio)
train_size = len(df) - test_size

# Вибір випадкових рядків для тестової і дослідної вибірок
test_data = df.sample(n=test_size)  # Вказати random_state для відтворюваності випадкових результатів
train_data = df.drop(test_data.index)
print(test_data)

# для дослідної
column_years = train_data['year'].tolist()
column_index = train_data['index'].tolist()


# для тестової
column_years_test = test_data['year'].tolist()
column_index_test = test_data['index'].tolist()


x_mean = np.mean(column_years)
y_mean = np.mean(column_index)



x_min, x_max = min(column_years), max(column_years)
y_min, y_max = min(column_index), max(column_index)

# Побудова графіку розсіювання
plt.scatter(x='year', y='index', data=train_data)

plt.scatter(x_mean, y_mean, color='red')


x_linspace = np.linspace(min(column_years), max(column_years), len(column_years))
y_linspace = np.linspace(min(column_index), max(column_index), len(column_index))

plt.ylim(y_min,y_max)
# sum of squared estimate of errors
sse = []
y_array = []
a_array = []
b_array = []
array = []  # для порівняння з тестовими
'''
a = math.tan(np.deg2rad(27))
b = y_mean - a * x_mean
Y_pred = a * np.array(column_years) + b

plt.plot(x_linspace, Y_pred)
print(Y_pred)

a = math.tan(np.deg2rad(46))
b = y_mean - a * x_mean
Y_pred = a * np.array(column_years) + b

plt.plot(x_linspace, Y_pred)
print(Y_pred)

a = math.tan(np.deg2rad(10))
b = y_mean - a * x_mean
Y_pred = a * np.array(column_years) + b

plt.plot(x_linspace, Y_pred)
print(Y_pred)

a = math.tan(np.deg2rad(73))
b = y_mean - a * x_mean
Y_pred = a * np.array(column_years) + b

plt.plot(x_linspace, Y_pred)
print(Y_pred)
'''
for k in range(90):
    a = math.tan(np.deg2rad(k))
    b = y_mean - a * x_mean
    Y_pred = a * np.array(column_years) + b
    y = math.tan(np.deg2rad(k)) * (x_linspace - x_mean) + y_mean
    #plt.plot(x_linspace, y)
    # Обчислення SSE    [ sigma ]
    sse.append(np.sum((np.array(column_index) - Y_pred) ** 2))
    #print(np.sum((np.array(column_index) - Y_pred) ** 2))
    y_array.append(y)
    a_array.append(a)
    b_array.append(b)

    #time.sleep(5)
'''
# least squares method
# x-x_mean
x_func = np.empty_like(column_years)
for i in range(len(column_years)):
    x_func[i] = column_years[i] - x_mean

# y-y_mean
y_func = np.empty_like(column_index)
for i in range(len(column_index)):
    y_func[i] = column_index[i] - y_mean

# (x-x_mean)^2
func_2 = np.empty_like(column_years)
for i in range(len(column_years)):
    func_2[i] = (column_years[i] - x_mean)**2


# (x-x_mean)*(y-y_mean)
func_multiplication = np.empty_like(column_years)
for i in range(len(column_years)):
    func_multiplication[i] = (column_years[i] - x_mean)*(column_index[i] - y_mean)

# sum of func_multiplication
'''

min_sse_indexes = np.argsort(sse)[:10]
#plt.plot(x_linspace, y_array[min_sse_indexes[0]], label=f'a={a_array[min_sse_indexes[0]]:.2f}, b={b_array[min_sse_indexes[0]]:.2f}, SSE={sse[min_sse_indexes[0]]:.2f}')

for i in range(len(min_sse_indexes)):
    plt.plot(x_linspace, y_array[min_sse_indexes[i]], label=f'a={a_array[min_sse_indexes[i]]:.2f}, b={b_array[min_sse_indexes[i]]:.2f}, SSE={sse[min_sse_indexes[i]]:.2f}')
    print("a:", a_array[min_sse_indexes[i]], " b:", b_array[min_sse_indexes[i]], " SSE:", sse[min_sse_indexes[i]])

plt.xlabel('Year')
plt.ylabel('Index')
plt.title('Graphic')

plt.legend(fontsize=8)

n = 10  # число для кількості прямих які я хочу вивести
# для тестової вибірки
sse_test = []
y_pred2 =[]
for i in range(n):
   for j in range(len(test_data)):
     y_pred2.append(a_array[i] * column_years_test[j] + b_array[i])
    #print(y_pred2[i]-column_index_test[i])
j = 0
error = np.zeros((10, 3))
for i in range(n):
 if i%3 == 0:
  j=0
 error[i,j]=y_pred2[i] - column_index_test[j]
'''
for i in range(len(min_sse_indexes)):
    #plt.plot(x_linspace, y_array[min_sse_indexes[i]], label=f'a={a_array[min_sse_indexes[i]]:.2f}, b={b_array[min_sse_indexes[i]]:.2f}, SSE={sse[min_sse_indexes[i]]:.2f}')
    print("a:", a_array[min_sse_indexes[i]], " b:", b_array[min_sse_indexes[i]], " Похибка:", sse_test[i])
#print(len(sse_test))
'''

Y_predicted = []
for i in range(n):
    #plt.plot(x_linspace, y_array[min_sse_indexes[i]], label=f'a={a_array[min_sse_indexes[i]]:.2f}, b={b_array[min_sse_indexes[i]]:.2f}, SSE={sse[min_sse_indexes[i]]:.2f}')
    Y_predicted.append([b_array[min_sse_indexes[i]] + a_array[min_sse_indexes[i]] * x for x in column_years_test])

    # Обчислення похибок

    #print("Похибки:", errors)


    #print("a:", a_array[min_sse_indexes[i]], " b:", b_array[min_sse_indexes[i]], " SSE:", sse[min_sse_indexes[i]])

for i in range(n):
    for j in range(len(test_data)):

        Y_predicted[i][j] = Y_predicted[i][j] - column_index_test[j]
# Застосування моделі до тестових даних для прогнозу

for i in range(n):
    print("a:", a_array[min_sse_indexes[i]], " b:", b_array[min_sse_indexes[i]], " Похибки:", Y_predicted[i])
plt.show()