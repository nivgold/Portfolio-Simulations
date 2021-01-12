import yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats

portfolio_stocks = ['VMC', 'EMR', 'CSX', 'UNP']
num_of_stocks = len(portfolio_stocks)
portfolio_weights = (0.25, 0.25, 0.25, 0.25)

def load_data():
    stocks_close = pd.DataFrame({})

    for stock_name in portfolio_stocks:
        ticker = yfinance.Ticker(stock_name)
        data = ticker.history(interval="1d",
                              start="1980-01-01", end="2018-03-26")

        data[f'{stock_name}'] = data['Close']

        stocks_close = stocks_close.join(data[[f'{stock_name}']],
                               how='outer').dropna()

    stocks_returns = get_returns_df(stocks_close)
    return stocks_returns

def get_returns_df(data):
    d = {}

    for stock_name in data.columns:
        stock_data = data[stock_name].values
        y = [100*((stock_data[i]/stock_data[i-1])-1) for i in range(1, len(stock_data))]
        d[stock_name] = y

    df = pd.DataFrame(d)
    df.index = data.index[1:]
    return df

def stocks_returns(data):

    sns.set_theme()

    for stock_name in data.columns:
        # stock_name = stock.replace("Close_", "")

        stock_data = data[stock_name].values
        x = np.arange(len(data.index))

        # calculating the stock correlation
        r = stats.pearsonr(x, stock_data)[0]

        mean = np.mean(stock_data)
        std = np.std(stock_data)

        # plotting histograms
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.histplot(stock_data, ax=ax)
        plt.xlim(-15, 15)
        plt.title(f"Returns of {stock_name} per day")
        plt.xlabel("Returns value (percentage)")
        plt.text(2.5, 300, r'$\mu={:.3f},\ \sigma={:.3f}$'.format(mean, std))
        plt.text(2.5, 250, r'$r={:.3f}$'.format(r))
        plt.grid(True)
        plt.savefig(f"visualizations/{stock_name}_histogram.png")
        plt.clf()

    # calculate the covariance matrix between the stocks
    covariance_matrix = data.cov()
    plt.figure(figsize=(8, 8))
    sns.heatmap(covariance_matrix, annot=True)
    plt.title("Covariance matrix between the stocks' returns", fontsize='17')
    plt.savefig("visualizations/covariance_matrix.png")

def get_small_window_days(data_days):
    data_days = np.array(data_days)

    num_forecast_days = 880
    small_window_size = 10
    num_of_small_window = int(num_forecast_days / small_window_size)

    small_windows = []

    random_dates = random.sample(range(len(data_days)-10), num_of_small_window)

    for date in random_dates:
        random_small_window = np.take(data_days, range(date, date+small_window_size))
        # random_small_window = data_days[date: date+small_window_size].copy()
        small_windows.append(random_small_window)

    artificial_window = []
    # flatten the small_windows list
    for l in small_windows:
        for elem in l:
            artificial_window.append(elem)

    return np.array(artificial_window)

def mult_agg(series):
    x = series.map(lambda x: x + 1)
    for i in x:
        print(i)
        break
    return reduce(lambda a, b: a*b, x)

def create_sliding_window_simulation(data, num_of_windows=100, deposit=False):

    data_values = data.values / 100

    data_days = np.arange(len(data))

    simulated_data = []

    for i in range(num_of_windows):
        small_window_days = get_small_window_days(data_days)
        num_of_days = len(small_window_days)

        small_window_data = data_values[small_window_days, :].copy()

        small_window_data = small_window_data.reshape(-1, num_of_stocks)
        small_window_data = small_window_data + 1

        aggregated = np.prod(small_window_data, axis=0)

        if deposit:
            aggregated = np.ones((num_of_stocks,))
            is_above_36 = [False for i in range(num_of_stocks)]

            for day in small_window_data:
                for stock_index in range(len(day)):
                    # checking that this stock has not reached above 36 yet
                    if not is_above_36[stock_index]:
                        daily_stock_return = day[stock_index]
                        # calculating the new stock return
                        aggregated[stock_index] = aggregated[stock_index] * daily_stock_return
                        if aggregated[stock_index] > 1.36:
                            # change this stock total value to 2%
                            aggregated[stock_index] = 1.02
                            # from now on we will not even check this stock because it reached above 36
                            is_above_36[stock_index] = True

            aggregated[aggregated < 1] = 1

        # aggregated is just one artificial window - just one sample in the simulation
        simulated_data.append(aggregated)

    simulated_data = pd.DataFrame(simulated_data, columns=data.columns)
    simulated_data = (simulated_data - 1) * 100
    simulated_data = simulated_data.applymap(lambda x: round(x, 3))
    simulated_data = simulated_data.astype(np.float32)

    return simulated_data

def calculate_portfolio_probabilities(simulation_data):
    # calculate the total portfolio return
    simulation_data['Total_Return'] = simulation_data['VMC']*0.25 + simulation_data['EMR']*0.25 + simulation_data['CSX']*0.25 + simulation_data['UNP']*0.25

    simulated_portfolio_return = simulation_data['Total_Return'].values
    n = len(simulated_portfolio_return)

    # print(np.sort(np.unique(simulated_portfolio_return)))

    # calculate probability for 0%
    print(100*'-')
    print("Probability of 0% Return:")
    prob_0 = (simulated_portfolio_return == 0.0).sum() / n
    print("{:.3f}".format(prob_0))

    print(100 * '-')
    print("Probability of 2% Return:")
    prob_2 = (simulated_portfolio_return == 2.0).sum() / n
    print("{:.3f}".format(prob_2))

    print(100 * '-')
    print("Probability of [20%, 2%) Return:")
    prob_20_2 = ((simulated_portfolio_return > 2.0) & (simulated_portfolio_return <= 20.0)).sum() / n
    print("{:.3f}".format(prob_20_2))

    print(100 * '-')
    print("Probability of (20%, 36%) Return:")
    prob_20_36 = ((simulated_portfolio_return > 20.0) & (simulated_portfolio_return < 36.0)).sum() / n
    print("{:.3f}".format(prob_20_36))

    print(100 * '-')
    print("Return's Mean:")
    mean = np.mean(simulated_portfolio_return)
    print("{:.3f}".format(mean))

    print(100 * '-')
    t = 1.660
    mean = np.mean(simulated_portfolio_return)
    std = np.std(simulated_portfolio_return)
    print("Return's Confidence Interval:")
    # percentile_10th = np.percentile(simulated_portfolio_return, 10)
    # percentile_90th = np.percentile(simulated_portfolio_return, 90)
    # print("P(" + str(round(percentile_10th, 3)) + " <= mu <= " + str(round(percentile_90th, 3)) + ") = 0.9")
    lower = mean - t * (std/(np.sqrt(n)))
    upper = mean + t * (std/(np.sqrt(n)))
    print("P(" + str(round(lower, 3)) + " <= mu <= " + str(round(upper, 3)) + ") = 0.9")

def ex_1():
    returns_data = load_data()
    stocks_returns(returns_data)

def ex_2():
    # -----------------a------------------
    print(49*"-"+"a."+49*"-")
    print("Simple Portfolio")
    returns_data = load_data()
    sim_data = create_sliding_window_simulation(returns_data, deposit=False)
    calculate_portfolio_probabilities(sim_data)

    # -----------------b------------------
    print(49 * "-" + "b." + 49 * "-")
    print("Market-linked CD")
    returns_data = load_data()
    sim_data = create_sliding_window_simulation(returns_data, deposit=True)
    calculate_portfolio_probabilities(sim_data)

    # -----------------c.1------------------
    print()
    print(49 * "-" + "c.1" + 48 * "-")
    print("Simple Portfolio")
    returns_data = load_data()
    sim_data = create_sliding_window_simulation(returns_data, num_of_windows=200, deposit=False)
    calculate_portfolio_probabilities(sim_data)

    # -----------------c.2------------------
    print()
    print(49 * "-" + "c.2" + 48 * "-")
    print("Market-linked CD")
    returns_data = load_data()
    sim_data = create_sliding_window_simulation(returns_data, num_of_windows=200, deposit=True)
    calculate_portfolio_probabilities(sim_data)

if __name__ == '__main__':
    ex_1()
    # ex_2()
