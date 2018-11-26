import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# set weights

def set_LTU(x, y, z):
    w0 = x
    w1 = y
    w2 = z
    return [w0, w1, w2]


def count_avg(data):
    data = np.array(data)
    l = []
    for i in range(0, len(data) - 1):
        l.append((float(data[i]) + float(data[i + 1])) / 2)
    return l


def count_score(data, ISET, avg):
    data = np.array(data)
    ISET = np.array(ISET['x0'])
    avg = np.array(avg)
    l = []

    for i in range(len(avg)):
        for j in range(len(data)):
            l.append((float(data[j]) * float(ISET[j])) - float(avg[i]))

    return np.array(l).reshape(len(avg), len(data))


def count_ratio(dataFrame):
    l = []
    for x in range(len(dataFrame)):
        for i in range(0, 2):
            if dataFrame.iloc[x][i] > 0:
                i += 1
        l.append(float(i))
        for j in range(0, 2):
            if dataFrame.iloc[x][j + 2] < 0:
                j += 1
        l.append(float(j))

    return np.array(l).reshape(3, 2).sum(axis=1) / 4


ISET = pd.read_csv('data.csv', delimiter=',')
ATTS = ['x1', 'x2']


def iwp(ISET, ATTS):
    h= set_LTU(-0.5, 1,6)
    best = h

    count = 0
    while count != 3:
        for k in range(len(ATTS)-1):
            ukj = sorted((h[k] * ISET['x1'] + h[k + 1] * ISET['x2']) / ISET['x0'], reverse=False)
            ukj1 = (h[k] * ISET['x1'] + h[k + 1] * ISET['x2']) / ISET['x0']

            print('----------' +'Iteration'+str(count+1)+'----------')
            data_array = pd.DataFrame(ukj)
            un_data = pd.DataFrame(ukj1)

            ISET['U0j'] = un_data
            ISET['U0j;'] = data_array

            print(ISET)

            avg = pd.DataFrame(count_avg(data_array))
            avg *= -1

            data = pd.DataFrame(ukj1)

            score_table = pd.DataFrame(count_score(data, ISET, avg), columns=['V+', 'Vi+', "V-", "Vi-"])

            score_table['Score'] = count_ratio(score_table)
            score_table['Avg'] = avg

            # print("Table with Score\n")

            print(score_table)

            score_table = score_table.sort_values(by='Score', ascending=False)

            v = score_table['Avg'].where(score_table['Score'] > score_table['Score'].min())
            v = v.dropna()

            print("w0:")
            print(v.min())

            ukj2 = sorted(((v.min() * ISET['x0'] + h[k + 1] * ISET['x2']) / ISET['x1']), reverse=True)
            ukj3 = (v.min() * ISET['x0'] + h[k + 1] * ISET['x2']) / ISET['x1']

            data_array1 = pd.DataFrame(ukj2)
            u_data = pd.DataFrame(ukj3)

            ISET['U0j'] = u_data
            ISET['U0j;'] = data_array1

            print(ISET)

            avg1 = pd.DataFrame(count_avg(data_array1))

            data1 = pd.DataFrame(ukj3)
            score_table1 = pd.DataFrame(count_score(-data1, ISET, avg1), columns=['V+', 'Vi+', "V-", "Vi-"])

            score_table1['Score'] = count_ratio(score_table1)
            score_table1['Avg'] = -avg1

            # print("Table with Score\n")

            print(score_table1)

            score_table1 = score_table1.sort_values(by='Score', ascending=False)

            v1 = score_table1['Avg'].where(score_table1['Score'] > score_table1['Score'].min())
            v1 = v1.dropna()
            print('w1:')
            print(v1.min())

            h = set_LTU(v1.min(), 1, v.min())
            print("LTU:")
            print(h)

            for a in range(len(score_table1)):
                if score_table1['Score'][a] == 1:
                    best = h
                    print("Best:")
                    print(best)

                    return best

        count += 1

iwp(ISET, ATTS)
