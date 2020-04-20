import numpy as np
import pandas as pd


class Model:
    def __init__(self, predictors, y, DataFrame, fix_order, var_ss_flag):
        self.multiple = False
        if len(predictors) > 1:
            self.multiple = True
        self.y_name = y
        self.predictors = predictors
        self.df = self._clean_(y, DataFrame)
        if fix_order:
            self.df_in_order = pd.DataFrame()
            for pred in predictors:
                self.df_in_order[pred] = self.df[pred]
            self.df_in_order[y] = self.df[y]
            self.df = self.df_in_order
        self.all_columns = self.df.drop(y, axis=1).columns
        to_drop = [col for col in self.all_columns if col not in predictors]
        if len(to_drop) != 0:
            self.df = self.df.drop(to_drop, axis=1)
        self.columns = self.df.drop(y, axis=1).columns
        self.x = self.df.drop(y, axis=1).values
        self.y = np.round(np.asarray(self.df[y].values), 4)

        # TODO: outliers
        self.assumptions = True  # TODO: [5 assumptions function defined above]
        if not self.assumptions:
            print("Assumptions failed")
            pass
        else:
            self.n = len(self.df)
            self.k = len(predictors)
            self.beta_hats_all = self._beta_hat_matrix_(self.x, self.y)
            self.beta_zero = self.beta_hats_all[0]
            self.beta_hats = self.beta_hats_all[1:]
            self.sse = self._get_ss_error_()
            self.sse_df = self.n - len(self.beta_hats_all)
            self.mse = np.round(self.sse/self.sse_df, 4)
            self.ssr = self._get_ss_regression_()
            self.sst = self.sse + self.ssr
            if self.multiple and var_ss_flag:
                self.var_ss_in_order = self._get_var_ss_()

    def __predict__(self, xvals):
        pred = self.beta_zero
        for x in range(len(xvals)):
            pred += xvals[x] + self.beta_hats[x]
        return pred

    def __str__(self):
        out = 'MODEL: '
        out += str(self.beta_zero)
        i = 1
        for beta_hat in self.beta_hats:
            out += " + X" + str(i) + '*' + str(beta_hat)
            i += 1
        out += '\n'
        out += '--------------------------------------\n'
        out += "SS[type]: df value\n"
        pretty_ssr = f"SS[regression]: {self.k}  {self.ssr}\n"
        out += pretty_ssr
        if self.multiple:
            pred_len = len(max(self.predictors, key=len))
            for i in range(len(self.var_ss_in_order)):
                ss = str('{:.4f}'.format(round(self.var_ss_in_order[i], 4)))
                out += str(str("SS[" + str(self.predictors[i]) + "]:") + " 1 " + str(ss) + '\n')
        pretty_sse = str(f"SS[error/residuals]: {self.sse_df}  {self.sse}\n")
        out += pretty_sse
        pretty_sst = str(f"SS[total]: {self.n - 1}  {self.sst}\n")
        out += pretty_sst
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
        if np.shape(X)[0] == 1:
            X = X.reshape(-1, 1)
        # add the one vector
        X = np.concatenate((np.ones(shape=X.shape[0]).reshape(-1, 1), X), 1)
        # beta hat matrix (0th indeth is B0) - will have len(2) minimum
        beta_hats = np.round(np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y), 4)
        return beta_hats

    def _get_ss_error_(self):
        y_hats = np.round(self._ss_regression_helper_(), 4)
        sse = 0
        for i in range(len(y_hats)):
            sse += (y_hats[i] - self.y[i])**2
        sse = np.round(sse, 4)
        return sse

    def _get_ss_regression_(self):
        y_hats = np.round(self._ss_regression_helper_(), 4)
        ssr = 0
        y_hat_bar = np.mean(y_hats)
        for i in range(len(y_hats)):
            ssr += (y_hats[i] - y_hat_bar)**2
        ssr = np.round(ssr, 4)
        return ssr

    def _ss_regression_helper_(self):
        preds = []
        for row in self.x:
            i = 0
            row_sum = 0
            for col in row:
                row_sum +=col*self.beta_hats[i]
                i += 1
            preds.append(row_sum + self.beta_zero)
        return preds

    def _get_var_ss_(self):
        ss, curr_preds = [], []
        i = 0
        for pred in self.predictors:
            curr_preds.append(pred)
            curr_model = Model(curr_preds, self.y_name, self.df, False, False)
            if i == 0:
                ss.append(curr_model.ssr)
                i += 1
                continue
            ss.append(curr_model.ssr - sum(ss))
            i += 1
        return ss

    #  TODO: all r values
    #  TODO: all hypothesis tests, formmated to work with size n predictors
    #  TODO: cofounding
    #  TODO: model optimizer (gets best models)
    #  TODO: model saved (saves models to be used)
