from framework import Model
import pandas as pd


def main():
    df = pd.read_csv(r"C:\Users\teamj\Documents\regretion_framework\test2.csv")
    x = ['AGE', 'QUET']
    x2 = ['QUET', 'AGE']
    lm = Model(x, 'SBP', df)
    print(lm)
    # print(lm.columns)


if __name__ == "__main__":
    main()
