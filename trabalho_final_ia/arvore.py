import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import pydotplus
dataset = pd.read_csv('training.csv')
genero = []
for i in dataset.index:
 if int(dataset["pop"][i]) == 1:
  # print(f'{i}--{dataset["pop"][i]}')
  genero.append(1)
 elif int(dataset["rock"][i]) == 1:
    # print(f'{i}--{dataset["rock"][i]}')
  genero.append(2)
 elif int(dataset["hip hop"][i]) == 1:
    # print(f'{i}--{dataset["hip hop"][i]}')
  genero.append(3)
 elif int(dataset["Dance/Electronic"][i]):
  #  print(f'{i}')
  genero.append(4)

dataset=dataset.assign(genero=genero)
print(dataset)
dataset = dataset.drop(columns=['pop','rock','hip hop','Dance/Electronic'])
print(dataset)
# dataset = pd.read_csv('wine.data', header = None)
#1:pop	2:rock	3:hip hop	4:Dance/Electronic

# dataset.columns = ['label',
#                    'alcohol', 
#                    'malic_acid', 
#                    'ash', 
#                    'alcalinity_of_ash', 
#                    'magnesium', 
#                    'total_phenols', 
#                    'flavanoids', 
#                    'nonflavanoid_phenols', 
#                    'proanthocyanins', 
#                    'color_intensity', 
#                    'hue',
#                    'OD280/OD315',
#                    'proline']

# print(f'{dataset}')
from sklearn.model_selection import train_test_split

x = dataset.values[:, 1:]
y = dataset.values[:, 0] # a primeira coluna do dataset indica a origem do vinho 

# print(f'{x}')
# print(f'{y}')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(f'{x_train}')
# print(f'{x_test}')

#treinamento
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_model(height):
  model = DecisionTreeClassifier(criterion = 'entropy', max_depth = height, random_state = 0)
  model.fit(x_train, y_train)
  return model


#avaliacao
for height in range(1, 21): # 1-20
  model = train_model(height)
  y_pred = model.predict(x_test)
  
  print('--------------------------------------------------------------\n')
  print(f'Altura - {height}\n')
  print("Precisão: " + str(accuracy_score(y_test, y_pred)))
  
  #exportar a arvore
from IPython.display import Image 
from sklearn.tree import export_graphviz

model = train_model(3)


feature_names = ['alcohol',
                 'malic_acid',
                 'ash',
                 'alcalinity_of_ash', 
                 'magnesium', 
                 'total_phenols', 
                 'flavanoids', 
                 'nonflavanoid_phenols', 
                 'proanthocyanins', 
                 'color_intensity', 
                 'hue',
                 'OD280/OD315',
                 'proline']

classes_names = ['%.f' % i for i in model.classes_]

dot_data = export_graphviz(model, filled=True, feature_names=feature_names, class_names=classes_names, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())
graph.write_png("tree.png")
Image('tree.png')