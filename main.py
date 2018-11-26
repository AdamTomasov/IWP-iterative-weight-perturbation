import IWP
import pandas as pd

if __name__ == "__main__":
    ISET = pd.read_csv('data.csv', delimiter=',')
    ATTS = ['x1', 'x2']

    IWP.iwp(ISET,ATTS)