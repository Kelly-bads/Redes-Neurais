from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, metrics
import numpy as np

db = datasets.load_digits()

x = db.data
y = db.target

np.random.seed(0)

n_samples = len(x)
particao = 0.75

order = np.random.permutation(n_samples)

x = x[order]
y = y[order]

x_treino = x[:int(n_samples * particao)]
y_treino = y[:int(n_samples * particao)]

x_teste = x[int(n_samples * particao):]
y_teste = y[int(n_samples * particao):]

clf = DecisionTreeClassifier(criterion='entropy',splitter='random')

clf.fit(x_treino, y_treino)

predicao = clf.predict(x_teste)

matriz = metrics.confusion_matrix(y_teste, predicao)

print(clf.score(x_teste, y_teste))

for linha in matriz:
    print(linha)
