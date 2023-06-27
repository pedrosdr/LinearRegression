"""
Module created by Pedro Sartori Dias dos Reis
GitHub: https://github.com/pedrosdr
"""

import pandas as pd
import numpy as np
from sklearn import metrics


class LinearRegression:
    def __init__(self, y: list, x_params: list):
        self.y = y
        self.x = x_params

    @property
    def B(self) -> list:
        x = np.array(self.x)

        df_x = pd.DataFrame()
        df_x['a'] = 1

        for i in range(len(x)):
            df_x[i] = x[i]

        df_x['a'] = 1

        X = df_x.to_numpy()
        Y = np.array(self.y)

        B = np.linalg.inv(X.T @ X) @ X.T @ Y
        return list(B)

    @property
    def Y_proc(self):
        B = self.B
        y_proc = []

        for i in range(len(self.x[0])):
            yi_proc = B[0]

            for j in range(len(self.x)):
                yi_proc += B[j+1] * self.x[j][i]

            y_proc.append(yi_proc)

        return y_proc

    @property
    def R2(self):
        B = self.B

        res = pd.DataFrame()
        res['y_true'] = self.y
        res['y_proc'] = self.Y_proc
        return metrics.r2_score(res['y_true'], res['y_proc'])

    def __str__(self):
        return f'Matrix B = {self.B}\nThe first value of the list (index=0) represents the constant of the function, ' \
               f'followed by the "x" values (x, x2, x3, ...) '


class PolynomialLinearRegression:
    def __init__(self, x: list, y: list, polynomial_degree: int):
        self.x = x
        self.y = y
        self.deg = polynomial_degree

    @property
    def B(self) -> list:
        x = np.array(self.x)

        df_x = pd.DataFrame()
        df_x['a'] = 1

        for i in range(self.deg):
            df_x[i] = x**(1+i)

        df_x['a'] = 1

        X = df_x.to_numpy()
        Y = np.array(self.y)

        B = np.linalg.inv(X.T @ X) @ X.T @ Y
        return list(B)

    @property
    def Y_proc(self):
        B = self.B
        y_proc = []
        for x in self.x:
            yi_proc = B[0]
            for i in range(self.deg):
                yi_proc += B[i + 1] * x ** (i + 1)

            y_proc.append(yi_proc)

        return y_proc

    @property
    def R2(self):
        B = self.B

        res = pd.DataFrame()
        res['y_true'] = self.y
        res['y_proc'] = self.Y_proc
        return metrics.r2_score(res['y_true'], res['y_proc'])

    def __str__(self):
        return f'Matrix B = {self.B}\nThe first value of the list (index=0) represents the constant of the function, ' \
               f'followed by the "x" values (x, x^2, x^3, ...) '
