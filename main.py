import yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    portfolio_composition = ['VMC', 'EMR', 'CSX', 'UNP']

    stocks_close = pd.DataFrame({})

    for stock_name in portfolio_composition:
        ticker = yfinance.Ticker(stock_name)
        data = ticker.history(interval="1d",
                              start="1980-01-01", end="2019-12-31")

        data[f'Close_{stock_name}'] = data['Close']

        stocks_close = stocks_close.join(data[[f'Close_{stock_name}']],
                               how='outer').dropna()

    return stocks_close

def stocks_returns(data):

    sns.set_theme()

    for stock in data.columns:
        stock_name = stock.replace("Close_", "")

        stock_data = data[stock].values
        x = np.arange(len(data.index))[1:]

        y = [100*((stock_data[i]/stock_data[i-1])-1) for i in range(1, len(stock_data))]

        # calculating the stock correlation
        r = np.corrcoef(x, y)[0, 1]

        mean = np.mean(y)
        std = np.std(y)

        # plotting histograms
        # plt.hist(y, bins=50, alpha=0.75, density=True, facecolor='g')
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.histplot(y, ax=ax)
        plt.xlim(-0.01, 15.01)
        plt.title(f"Returns of {stock_name} per day")
        plt.xlabel("Returns value (percentage)")
        plt.text(6, 300, r'$\mu={:.3f},\ \sigma={:.3f}$'.format(mean, std))
        plt.text(6, 250, r'$r^2={:.3f}$'.format(r))
        plt.grid(True)
        plt.savefig(f"visualizations/{stock_name}_histogram.png")
        plt.clf()

    # calculate the covariance matrix between the stocks
    covariance_matrix = data.cov()
    plt.figure(figsize=(8, 8))
    sns.heatmap(covariance_matrix, annot=True)
    plt.title("Covariance matrix between the stocks' returns", fontsize='17')
    plt.savefig("visualizations/covariance_matrix.png")

def ex_1():
    data = load_data()
    stocks_returns(data)

if __name__ == '__main__':
    ex_1()