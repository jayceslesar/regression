import numpy as np
import pandas as pd


class Model:
    def __init__(self, predictors, y, DataFrame):
        self.cleaned_data = self._clean_(y, DataFrame)
        cols = self.cleaned_data.drop(y, axis=1).columns
        to_drop = [col for col in cols if col not in predictors]
        if len(to_drop) != 0:
            self.cleaned_data = self.cleaned_data.drop(to_drop, axis=1)
        self.x = self.cleaned_data.drop(y, axis=1).values
        self.y = np.round(np.asarray(self.cleaned_data[y].values), 4)

        # TODO: outliers
        self.assumptions = True  # TODO: [5 assumptions function defined above]
        if not self.assumptions:
            print("Assumptions failed")
        else:
            self.n = self.y.size
            self.k = len(predictors)
            self.beta_hats_all = self._beta_hat_matrix_(self.x, self.y)
            self.beta_zero = self.beta_hats_all[0]
            self.beta_hats = self.beta_hats_all[1:]
            self.ssr = self._get_ss_resid_()

    def __str__(self):
        print("SSR: " + str(self.ssr))
        out = str(self.beta_zero)
        i = 1
        for beta_hat in self.beta_hats:
            out += " + X" + str(i) + '*' + str(beta_hat)
            i += 1
        return out

    def _clean_(self, y, DataFrame):
        cleaned = DataFrame
        # drop all rows where y does not have a value
        zeroes = [num[0] for num in np.argwhere(cleaned[y].values == 0)]
        cleaned = cleaned.drop(zeroes)
        # fill in remaining NA's as 0's (get a better dataset doofus)
        cleaned = cleaned.fillna(0)
        return cleaned

    def _beta_hat_matrix_(self, X, y):
        # one predictor case
        if len(X) == 1: return X.reshape(-1, 1)
        # add the one vector
        X = np.concatenate((np.ones(shape=X.shape[0]).reshape(-1, 1), X), 1)
        # beta hat matrix (0th indeth is B0) - will have len(2) minimum
        beta_hats = np.round(np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y), 4)
        return beta_hats

    def _get_ss_resid_(self):
        y_hats = np.round(self._predict_for_ss(), 4)
        ssr = 0
        for i in range(len(y_hats)):
            ssr += (y_hats[i] - self.y[i])**2
        ssr = np.round(ssr, 4)
        return ssr

    def _predict_for_ss(self):
        preds = []
        for row in self.x:
            i = 0
            row_sum = 0
            for col in row:
                row_sum +=col*self.beta_hats[i]
                i += 1
            preds.append(row_sum + self.beta_zero)
        return preds




    #  TODO: matrix represenation probably
    #  TODO: all r values
    #  TODO: all hypothesis tests, formmated to work with size n predictors
    #  TODO: cofounding
    #  TODO: model optimizer (gets best models)
    #  TODO: model saved (saves models to be used)
