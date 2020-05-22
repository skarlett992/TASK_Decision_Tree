import pandas as pd
import numpy as np
#from sortedcontainers import SortedDict

class Vertex:
    def __init__(self, x, y, depth, feature_index):
        self.x, self.y, self.depth, self.feature_index = x, y, depth, feature_index
        self.left, self.right = None, None
        self.best_col, self.best_val = None, None


        if self.depth < max_depth:
            self.build_subtree()

    @property
    def features(self):
        return DecisionTree.ginies_nat

    @property
    def probas(self):
        counts = self.y.value_counts(normalize=True).values
        return counts

    def build_subtree(self):
        best_gini, best_col, best_val = float('inf'), None, None
        col = self.features[self.feature_index]

        unique_values = np.nan_to_num(self.x[col].unique(), nan=float('inf'))
        unique_values.sort()
        len_unique = len(unique_values)
        value_each_person = self.x[col]

        for cur_val in unique_values:
            if len_unique > 4:
                left_yy = self.y[value_each_person <= cur_val]
                right_yy = self.y[value_each_person > cur_val]
            else:
                left_yy = self.y[value_each_person == cur_val]
                right_yy = self.y[value_each_person != cur_val]

            #подаем данные о нолях и единицах направо и налево, чтобы найти лучший джини текущий
            cur_gini = DecisionTree.gini(left_yy, right_yy)

            #после того, как он вычислил джини, смотрит, лучший ли он
            if cur_gini < best_gini:
                best_gini = cur_gini
                best_col = col
                best_val = cur_val
        #self.ginies_nat.append([best_gini, best_col])



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

            self.left = Vertex(left_x, left_y, self.depth + 1, self.feature_index+1)
            self.right = Vertex(right_x, right_y, self.depth + 1, self.feature_index+1)


    def get_next_vertex(self, x):
        if x[self.best_col] > self.best_val and self.right is not None:
            return self.right
        else:
            return self.left


class DecisionTree:
    def __init__(self):
        self.tree = None
    ginies_nat = []

    def fit(self, x, y):
        self._index = 0
        self.tree = Vertex(x, y, depth=0, feature_index=0)


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


    def order_features(self, x, y, features):
        for col in features:
            best_gini = float('inf')
            unique_values = np.nan_to_num(x[col].unique(), nan=float('inf'))
            unique_values.sort()
            len_unique = len(unique_values)
            value_each_person = x[col]

            for cur_val in unique_values:
                if len_unique > 4:
                    left_yy = y[value_each_person <= cur_val]
                    right_yy = y[value_each_person > cur_val]
                else:
                    left_yy = y[value_each_person == cur_val]
                    right_yy = y[value_each_person != cur_val]

                # подаем данные о нолях и единицах направо и налево, чтобы найти лучший джини текущий
                cur_gini = self.gini(left_yy, right_yy)

                # после того, как он вычислил джини, смотрит, лучший ли он
                if cur_gini < best_gini:
                    best_gini = cur_gini
            DecisionTree.ginies_nat.append([best_gini, col])
        DecisionTree.ginies_nat = sorted(DecisionTree.ginies_nat, key=lambda i: i[0])
        DecisionTree.ginies_nat = list(map(lambda x: x[1], DecisionTree.ginies_nat))

max_depth = 8
df = pd.read_csv('train.csv')
x, y = df[['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']], df['Survived']
tree = DecisionTree()
tree.order_features(x, y, features=x.columns)
tree.fit(x, y)

# ginies_all = dict(Vertex.ginies_nat)
# from sortedcontainers import SortedDict
# ginies_sort = list(SortedDict(ginies_all).items())
# DecisionTree.ginies = ginies_sort

for i in range(len(x)):
    test0 = x.iloc[i]
    test1 = tree.predict_proba(test0)
    if test1.size<2:
        continue
    test2 = test1[0]
    test3 = test1[1]



    #print(DecisionTree.ginies)
    print(f'Шанс на выживание: на {test3} - да и на {test2}  - нет')
   # print(ginies_all)

