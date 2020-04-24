from framework import Model
import pandas as pd


def main():
    df = pd.read_csv(r"C:\Users\teamj\Documents\regretion_framework\test2.csv")
    x1 = ['AGE']
    x1x2 = ['AGE', 'QUET']
    x1x2x3 = ['AGE', 'QUET', 'SMK']
    x1x3x3 = ['AGE', 'SMK', 'QUET']
    y = 'SBP'
    lm = Model(x1x2x3, y, df, True, True)
    lm = lm.autobest()
    print(lm)


if __name__ == "__main__":
    main()
