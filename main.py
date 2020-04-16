from framework import Model
import pandas as pd


def main():
    df = pd.read_csv(r"C:\Users\teamj\Documents\regretion_framework\test2.csv")
    x = ['AGE', 'QUET']
    print(Model(x, 'SBP', df))


if __name__ == "__main__":
    main()
