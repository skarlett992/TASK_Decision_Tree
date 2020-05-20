import pandas as pd
import numpy as np
max_depth = 6

class Vertex:
    def __init__(self, x, y, depth):
        self.x, self.y, self.depth = x, y, depth
        self.left, self.right = None, None
        self.best_col, self.best_val = None, None

        if self.depth < max_depth:
       #     if self.best_col is not None and self.best_val is not None:
            self.build_subtree()


        if self.depth == max_depth-1:
           self.left = None
           self.right = None


    @property
    def features(self):
        return self.x.columns

    @property
    def probas(self):
        counts = self.y.value_counts(normalize=True).values
        return counts

    def build_subtree(self):
        best_gini, best_col, best_val = float('inf'), None, None
        for col in self.features:
            unique_values = np.nan_to_num(self.x[col].unique(), nan=float('inf'))
            unique_values.sort()
            for cur_val in unique_values:
                tmp1 = self.x[col] <= cur_val
                left = self.y[tmp1]

                tmp2 = self.x[col] > cur_val
                right = self.y[tmp2]

                cur_gini = DecisionTree.gini(left, right)

                if cur_gini < best_gini:
                    # if col is not None and  cur_val is not None:
                    best_gini = cur_gini
                    best_col = col
                    best_val = cur_val

        x_gini = self.x[best_col]

        # if left.size > 0:
        x_left = self.x[x_gini <= best_val]
        y_left = self.y[x_gini <= best_val]
        self.left = Vertex(x_left, y_left, self.depth + 1)
        # if right.size > 0:
        x_right = self.x[x_gini > best_val]
        y_right = self.y[x_gini > best_val]
        self.right = Vertex(x_right, y_right, self.depth + 1)
        self.best_col, self.best_val = best_col, best_val

    def get_next_vertex(self, x):
        if x[self.best_col] > self.best_val and self.right is not None:
            return self.right
        else:
            return self.left


class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, x, y):
        self.tree = Vertex(x, y, depth=0)

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



df = pd.read_csv('train.csv')
x, y = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], df['Survived']
tree = DecisionTree()
tree.fit(x, y)

for i in range(20):
    test0 = x.iloc[i]
    test1 = tree.predict_proba(test0)
    if test1.size<2:
        continue
    test2 = test1[1]
    print(test2)
