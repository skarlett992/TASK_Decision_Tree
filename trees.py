import pandas as pd
import numpy as np


class Vertex:
    def __init__(self, x, y, depth, feature_index):
        self.x, self.y, self.depth, self.feature_index = x, y, depth, feature_index
        self.left, self.right = None, None
        self.best_col, self.best_val = None, None
        self.dict_ginies = {'Pclass': None, 'SibSp': None, 'Parch': None, 'Fare': None, 'Age': None}
        self.ginies = []

        if self.depth < max_depth:
            self.build_subtree()


    @property
    def features(self):
        return self.x.columns

    @property
    def probas(self):
        counts = self.y.value_counts(normalize=True).values
        return counts

    def build_subtree(self):
        best_gini, best_col, best_val = float('inf'), None, None
        col = self.features[self.feature_index]

        #for feat in col:

        unique_values = np.nan_to_num(self.x[col].unique(), nan=float('inf'))
        unique_values.sort()
        len_unique = len(unique_values)
        value_each_person = self.x[col]

        for cur_val in unique_values:
            if len_unique > 4:
                left_xx = self.x[value_each_person <= cur_val]
                left_yy = self.y[value_each_person <= cur_val]

                right_xx = self.x[value_each_person > cur_val]
                right_yy = self.y[value_each_person > cur_val]
            else:
                left_xx = self.x[value_each_person == cur_val]
                left_yy = self.y[value_each_person == cur_val]

                right_xx = self.x[value_each_person != cur_val]
                right_yy = self.y[value_each_person != cur_val]

            #подаем данные о нолях и единицах направо и налево, чтобы найти лучший джини текущий
            cur_gini = DecisionTree.gini(left_yy, right_yy)

            #после того, как он вычислил джини, смотрит, лучший ли он
            if cur_gini < best_gini:
                best_gini = cur_gini
                best_col = col
                best_val = cur_val



        self.make_leaves(value_each_person, best_val, len_unique)
        self.best_col, self.best_val = best_col, best_val

    # метод, создающий левый и правый узлы
    def make_leaves(self, value_each_person, best_val, len_unique):
        if self.feature_index < len(self.features)-1:
            if len_unique > 4:
                left_x = self.x[value_each_person <= best_val]
                left_y = self.y[value_each_person <= best_val]
                right_x = self.x[value_each_person > best_val]
                right_y = self.y[value_each_person > best_val]
            else:
                left_x = self.x[value_each_person == best_val]
                left_y = self.y[value_each_person == best_val]
                right_x = self.x[value_each_person != best_val]
                right_y = self.y[value_each_person != best_val]

            self.left = Vertex( left_x, left_y, self.depth + 1, self.feature_index+1)
            self.right = Vertex( right_x, right_y, self.depth + 1, self.feature_index+1)
           # self.dict_ginies[self.best_col] = self.best_gini

    def get_next_vertex(self, x):
        if x[self.best_col] > self.best_val and self.right is not None:
            return self.right
        else:
            return self.left


class DecisionTree:
    def __init__(self):
        self.tree = None
        self.ginies = []


    def fit(self, x, y):
        self._index = 0
        #создать новое условие - очередность джини
        #self.dict_ginies[self.best_col] = self.best_gini
       # self.features = sorted(self.dict_ginies.items(), key=lambda kv: kv[1])
        self.tree = Vertex(x, y, depth=0, feature_index=0)

    #def get_best_gini(self):


    def predict_proba(self, x):
        cur_vertex = self.tree
        while cur_vertex.left is not None:
            cur_vertex = cur_vertex.get_next_vertex(x)
        return cur_vertex.probas

    @staticmethod
    def gini(*xn):
        xn = list(xn)
        for i in range(len(xn)):
            if not isinstance(xn[i], pd.Series):
                xn[i] = pd.Series(xn[i])
        total = sum(x.shape[0] for x in xn)
        result = sum((1 - (x.value_counts(normalize=True)**2).sum()) * x.shape[0]/total for x in xn)
        if result==0:
            result = 0.01
        return result


max_depth = 8
df = pd.read_csv('train.csv')
x, y = df[['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']], df['Survived']
tree = DecisionTree()

# order_features - метод который выстроит фичи в порядке возрастания гини - сортировать словарь здесь

tree.fit(x, y)

for i in range(20):
    test0 = x.iloc[i]
    test1 = tree.predict_proba(test0)
    if test1.size<2:
        continue
    test2 = test1[1]
    print(test2)
