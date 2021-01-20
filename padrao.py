import random
import csv

datas = []
with open('dataset_net.csv') as _file:
    dado = csv.reader(_file, delimiter = ',')
    for linha in dado:
        linha = [float(item) for item in linha]
        datas.append(linha)

def tre_tes_split(dataset, porcentagem):

    percent = porcentagem * len(dataset) // 100
    dado_treino = random.sample(dataset, percent)
    dado_teste = [data for data in dataset if data not in dado_treino]

    def prepare(dataset):
        x, y = [], []
        for data in dataset:
            x.append(data[1:3])
            y.append(data[0])
        return x, y

    x_train, y_train = prepare(dado_treino)
    x_test, y_test = prepare(dado_teste)
    return x_train, y_train, x_test, y_test

def sign(u):

    return 1 if u >= 0 else -1

def ajuste(w, x, d, y):

    t_aprendiz = 0.01
    return w + t_aprendiz * (d - y) * x

def perceptron_f(x, d):

    ep = 0
    w = [random.random() for i in range(3)]
    print(w)
    while True:
        erro = False
        for i in range(len(x)):
            u = sum([w[0]*-1, w[1]*x[i][0], w[2]*x[i][1]])
            y = sign(u)
            if y != d[i]:
                w[0] = ajuste(w[0], -1, d[i], y)
                w[1] = ajuste(w[1], x[i][0], d[i], y)
                w[2] = ajuste(w[2], x[i][1], d[i], y)
                erro = True
        ep += 1
        if erro is False or ep == 1000:
            break
    print(ep)
    return w

def perceptron_pred(x_teste, w_ajustado):

    y_predict = []
    for i in range(len(x_teste)):
        pred = sum([w_ajustado[0]*-1, w_ajustado[1]*x_teste[i][0],
                       w_ajustado[2]*x_teste[i][1]])
        y_predict.append(sign(pred))
    return y_predict

def acuracia(y_teste, y_valid):

    total = 0
    for i in range(len(y_teste)):
        if y_teste[i] == y_valid[i]:
            total += 1
        else:
            pass
    return total / len(y_valid)

x_tre, y_tre, x_tes, y_tes = tre_tes_split(datas, 80)

w_fit = perceptron_f(x_tre, y_tre)
print(w_fit)

y_validado = perceptron_pred(x_tes, w_fit)
print(y_validado)

accuracy = acuracia(y_tes, y_validado)
print(accuracy)