import pandas as pd
import numpy as np
import pandas_talib as pta
import packageTFG.ta_functions as ta
import itertools
import re
from pprint import pprint
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt

pta.SETTINGS.join = False

base = 2

# Create periods list, to compute technical analysis over different periods
periods = list(set([np.int(np.round(base ** power))
                    for power in np.linspace(1, 8, 15)]))
periods.sort()
# Creates list [2, 3, 4, 5, 7, 9, 12, 16, 22, 29, 39, 53, 71, 95, 128]
# periods = [30, 50]


ta_periods = list(set([np.int(np.round(base ** power))
                    for power in np.linspace(1, 7, 6)]))
ta_periods.sort()
# [2, 5, 14, 37, 97, 256]

# Create target periods list, to compute returns n periods into the future
target_periods_list = list(set([np.int(np.round(base ** power))
                           for power in np.linspace(0, 5, 15)]))
target_periods_list.sort()
target_periods_list = [1]


variation_periods = list(set([np.int(np.round(base ** power))
                    for power in np.linspace(0, 5, 3)]))
variation_periods.sort()
# [1, 6, 32]



periods_MACD_short = list(set([np.int(np.round(base ** power))
                               for power in np.linspace(4, 7, 6)]))
periods_MACD_short.sort()
# Creates list [16, 28, 49, 84, 147, 256]
periods_MACD = list(itertools.combinations(periods_MACD_short, 2))
# Creates List
# [(16, 24),
#  (16, 37),
#  (16, 56),
#  (16, 84),
#  (16, 128),
#  (24, 37),
#  (24, 56),
#  (24, 84),
#  (24, 128),
#  (37, 56),
#  (37, 84),
#  (37, 128),
#  (56, 84),
#  (56, 128),
#  (84, 128)]


class Settings(object):
    join=False

SETTINGS=Settings()

functions = ['RET',
             'compute_function_different_periods',
             'nondimensionalize',
             'trade_vol_VS_tx_vol',
             'miners_revenue_VS_tx_volume',
             'block_reward_USD',
             'tx_fees_VS_miners_revenue',
             'velocity',
             'NVT_ratio',
             'avg_tx_fees_USD',
             'avg_tx_fees_BTC',
             'avg_tx_value_USD',
             'avg_tx_value_BTC'
             'fee_VS_tx_value',
             'add_derived_blockchain_data'
             ]


derived_blockchain_functions = ['trade_vol_VS_tx_vol',
                                'miners_revenue_VS_tx_volume',
                                'block_reward_USD',
                                'tx_fees_VS_miners_revenue',
                                'avg_tx_fees_USD',
                                'avg_tx_fees_BTC',
                                'avg_tx_value_USD',
                                'avg_tx_value_BTC',
                                'fee_VS_tx_value',
                                'velocity',
                                'NVT_ratio'
                                ]

columns_to_transform_log =  ['Close',
                         'High',
                         'Low',
                         'Open',
                         'Volume (BTC)',
                         'Volume (USD)',
                         'Tx fees (BTC)',
                         'Tx fees (USD)',
                         'Revenue per tx (USD)',
                         'Miners Revenue (USD)',
                         'Avg Block Size (MB)',
                         'Blockchain Size (GB)',
                         'Hash Rate (GH)',
                         'Exchange Trade Volume (USD)',
                         'Avg TXs per block',
                         'Blockchain.info wallet users',
                         'Unique Adresses Used per day',
                         'Txs (exl popular addresses)',
                         'Txs',
                         'Total txs',
                         'Tx Volume (BTC)',
                         'Tx Volume (USD)',
                         'Output Volume (USD)',
                         'Total BTC mined',
                         'Market Cap (USD)',
                         'Mempool tx count',
                         'Mempool Growth Rate (bytes/day)',
                         'Mempool Size (bytes)',
                         'UTXO count',
                         'Difficulty',
                         'Trade Vol / Tx Vol',
                         'Miners Revenue / Tx Volume',
                         'Block Reward (USD)',
                         'Tx Fees / Miners Revenue',
                         'Avg Tx Fees (USD)',
                         'Avg Tx Fees (BTC)',
                         'Avg Tx Value (USD)',
                         'Avg Tx Value (BTC)',
                         'Tx Fees / Tx Volume',
                         'Velocity',
                         'NVT Ratio',
                         'Volume',
                         'ATR',
                         'BollingerB',
                         'KelChM',
                         'KelChU',
                         'KelChD',
                         'STD',
                         'EMA']

columns_to_divide = ['ATR',
                     'Force',
                     'KelCh',
                     'STD',
                     'Momentum',
                     'MACD']


def out(settings, df, result):
    """Returns the result of a function that creates a new pandas.Series
    from the df as the original df with a new column or as Series

    Parameters
    ----------
        df : pandas.Dataframe
            original Dataframe
        result : pandas.Series or pandas.Dataframe
        settings : object of the type SETTINGS

    returns
    ----------
        settings.join is False : pandas.Series
        settings.join is True : pandas.DataFrame
    """
    if settings.join:
        return df.join(result[result.columns.difference(df.columns)])
    return result.squeeze()


def concat_without_duplicates(dfs):
    """Concatenate pandas.DataFrames that may have duplicate rows

    Parameters
    ----------
    dfs : list
        list of pandas.DataFrames

    Returns
    ----------
    concatenated : pandas.DataFrame
    """
    temp_dfs = []
    for temp_df in dfs:
        # Joining the different dfs resulted in a df with more rows. This is why
        # I do this. More info on https://stackoverflow.com/a/34297689/5031446
        # This removes rows with duplicated indexes and keeps just the last observation
        temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
        temp_dfs.append(temp_df)
    result = pd.concat(temp_dfs, axis=1)

    return result


def compute_derived_blockchain_data(df):
    """Returns data derived from the basic blockchain DataFrame"""

    original_join_state = SETTINGS.join

    SETTINGS.join = False
    result = pd.concat([trade_vol_VS_tx_vol(df),
                       miners_revenue_VS_tx_volume(df),
                       block_reward_USD(df),
                       tx_fees_VS_miners_revenue(df),
                       avg_tx_fees_USD(df),
                       avg_tx_fees_BTC(df),
                       avg_tx_value_USD(df),
                       avg_tx_value_BTC(df),
                       fee_VS_tx_value(df)], axis=1)

    velocity_df = compute_function_different_periods(df,
                                                     periods=ta_periods,
                                                     function=velocity)
    NVT_df = compute_function_different_periods(df,
                                                periods=ta_periods,
                                                function=NVT_ratio)

    result = pd.concat([result, velocity_df, NVT_df], axis=1)

    SETTINGS.join = original_join_state
    return out(SETTINGS, df, result)

def process_all_data(df,
                     blockchain_data=False,
                     verbose=True,
                     verbose_periods=False,
                     ta_periods=ta_periods,
                     macd_periods=periods_MACD,
                     variat_periods=variation_periods,
                     target_periods=None,
                     compute_calendar_data=False,
                     return_non_dropped=False,
                     decimals_time=3):
    """Process a dataframe containing OHLCV and data from blockchain.info

    Parameters
    ----------
    df : pandas.DataFrame
        containing OHLC data. Blockchain data optional. If Blockchain data
            not provided, just comment line derived_blockchain_data = ...
    verbose : boolean, default True
        print info of what's being done
    ta_periods : list of integers
    macd_periods : list of tuples (short period), (long period)
    variat_periods : list of integers
    target_periods : list of integers, default None
        if None, only X data is returned. If not none, Y data is returned too,
        containing returns to be calculated periods in teh future in target_periods
    return_non_dropped : bool, default False
        returns a Dataframe without dropping the NaN values in the last step
    blockchain_data : bool, default true
        computes data from blockchain.info or not


    Returns
    ----------
    result : pandas.DataFrame
        X data we can train the ML models with
    targets : pandas.DataFrame or Series
        Y data to train the ML models with"""


    start = time.process_time()

    original_join_state = SETTINGS.join
    SETTINGS.join = False

    result = df
    print('Original DataFrame shape: ', result.shape)
    print()

    # Replace outliers in volume data
    t = time.process_time()
    vol_cols = cols_in_df(result, ['Vol'])
    result[vol_cols] = result[vol_cols].apply(replace_outliers)
    elapsed_time = time.process_time() - t
    if verbose:
        print('Replace outliers in volume data')
        print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
              prec=decimals_time), result.shape)
        print()

    # Add blockchain derived data
    if blockchain_data:
        t = time.process_time()
        derived_blockchain_data = compute_derived_blockchain_data(df)
        result = concat([result, derived_blockchain_data])
        elapsed_time = time.process_time() - t
        if verbose:
            print('Add blockchain derived data')
            print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
                  prec=decimals_time), result.shape)
            print()

    # Add technical analysis indicators
    t = time.process_time()
    technical_analysis_df = technical_analysis(result, periods=ta_periods)
    result = concat_without_duplicates([result, technical_analysis_df])
    elapsed_time = time.process_time() - t
    if verbose:
        print('Add  technical analysis indicators')
        print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
              prec=decimals_time), result.shape)
        if verbose_periods:
            print('TA periods: ', ta_periods)
            print('MACD periods:')
            pprint(periods_MACD)
        print()

    # Calculate the log of a lot of the previous data
    t = time.process_time()
    cols_to_log_transform = cols_in_df(df=result,
                                       partial_col_names=columns_to_transform_log,
                                       not_present=['MACD', 'Force'])
    log_transformed_df = log_transform(result, cols=cols_to_log_transform)
    result = concat_without_duplicates([result, log_transformed_df])
    elapsed_time = time.process_time() - t
    if verbose:
        print('Calculate the log of a lot of the previous data')
        print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
              prec=decimals_time), result.shape)
        print()


    # Calculate radios dividing data by price, or viceversa. Compute drawdown functions
    t = time.process_time()
    mayer_ms = compute_mayer_multiples(result)
    drawdowns = compute_function_different_periods(result,
                                                   periods=ta_periods,
                                                   function=compute_derived_drawdown_functions)
    drawdown = compute_derived_drawdown_functions(df)
    ratios = compute_ratios_ta(result)
    dfs =  result, mayer_ms, ratios, drawdowns, drawdown
    result = concat_without_duplicates(dfs)
    elapsed_time = time.process_time() - t
    if verbose:
        print('Calculate ratios dividing data by price, or viceversa. Compute drawdown functions')
        print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
              prec=decimals_time), result.shape)
        print()


    # Computes velocities and accelerations  for all columns in the df
    t = time.process_time()
    # # ONE CORE
    # vel_acc = compute_vel_acc(result)

    # MULTI-CORE
    vel_acc = multiprocess_pandas_dataframe(func=vel_acc_series,
                                            df=result,
                                            periods=variation_periods)
    result = result.join(vel_acc)
    elapsed_time = time.process_time() - t
    if verbose:
        print('Compute velocity and acceleration (geom and simple) for all of the columns in the df')
        print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
              prec=decimals_time), result.shape)
        print()


    # Computes ROCs for all columns in the df
    # t = time.process_time()
    # print('Start computing variations')
    # variations = compute_variations(result, log=True, periods=variat_periods)
    # print('Variations computed')
    # result =  concat_without_duplicates([result, variations])
    # print('dfs concatenated')
    # elapsed_time = time.process_time() - t
    # if verbose:
    #     print('Computes variations (momentum or returns) for all of the columns in the df')
    #     print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
    #           prec=decimals_time), result.shape)
    #     print()


    # Forward fill the NaNs if there's any (improbable)
    t = time.process_time()
    # # ONE CORE
    # result = result.apply(replaces_nans_ma)
    # MULTI-CORE
    result = multiprocess_pandas_dataframe(func=replaces_nans_ma, df=result)
    elapsed_time = time.process_time() - t
    if verbose:
        print('Substitutes NaN, -inf and inf with the average of all the previous values')
        print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
              prec=decimals_time), result.shape)
        print()

    # Compute calendar ta_functions
    if compute_calendar_data:
        t = time.process_time()
        result = compute_calendar_functions(result)
        elapsed_time = time.process_time() - t
        if verbose:
            print('Compute calendar functions')
            print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
                  prec=decimals_time), result.shape)
            print()

    # Create target column(s)
    if target_periods:
        targets = compute_Y(df,
                            periods=target_periods,
                            function=compute_target,
                            log=True)
        result = result.join(targets)
        if verbose:
            print(result.shape, 'Create target DataFrame')

    # Drops the rows with any NaN value
    t = time.process_time()
    non_dropped = result
    result = result.dropna(axis=0)
    elapsed_time = time.process_time() - t
    if verbose:
        print('Drops  rows with any NaN value')
        print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
              prec=decimals_time), result.shape)
        print()


    # Separate result into X and Y for the ML tasks
    if target_periods:
        t = time.process_time()
        x_cols = result.columns.difference(targets.columns)
        X = result[x_cols]
        Y = result[targets.columns]
        elapsed_time = time.process_time() - t
        if verbose:
            print('Create X and Y for ML purposes')
            print("Time: {:.{prec}f} s. Shapes: ".format(elapsed_time,
                  prec=decimals_time), X.shape, Y.shape)
            print()

    else:
        t = time.process_time()
        X = result
        elapsed_time = time.process_time() - t
        if verbose:
            print('Create X for ML purposes')
            print("Time: {:.{prec}f} s. Shape: ".format(elapsed_time,
                  prec=decimals_time), result.shape)
            print()


    SETTINGS.join = original_join_state

    total_elapsed_time = time.process_time() - start
    print('Data processing completed.')
    print("Total time: {:.{prec}f} s. Shape: ".format(total_elapsed_time,
          prec=decimals_time), result.shape)
    print()


    if return_non_dropped:
        if target_periods:
            return non_dropped, X, Y
        else:
            return non_dropped, X
    else:
        if target_periods:
            return X, Y
        else:
            return X

def count_quadrants(target, predictions, threshold=0, return_quadrants_df=False) :
    """Counts the points that are in each quadrant in a scatter plot where target
    is on the x axis and predictions is on the y axis, separated by threshold

    Parameters:
    ----------
    target : pandas.Series
    predictions : pandas.Series
    threshold : numeric, deefault 0
    return_quadrants : bool, default False
        return the dataframes containint targets and predictions in each quadrant, or not

    Returns
    ----------
    sizes : list of integers
        number of the points present in each quadrand
    quadrants : list of pandas.DataFrames
        containing the DataFrames of the points present in each one of the quadrants
    """

    comparison = pd.concat([target, predictions], axis=1)
    comparison.columns = ['Target', 'Prediction']


    quadrant1 = comparison.loc[(comparison.Target < threshold ) & (comparison.Prediction > threshold)]
    quadrant2 = comparison.loc[(comparison.Target > threshold ) & (comparison.Prediction > threshold)]
    quadrant3 = comparison.loc[(comparison.Target < threshold ) & (comparison.Prediction < threshold)]
    quadrant4 = comparison.loc[(comparison.Target > threshold ) & (comparison.Prediction < threshold)]

    quadrants = [quadrant1,
                 quadrant2,
                 quadrant3,
                 quadrant4]

    sizes = [len(q) for q in quadrants]
    total_size = np.sum(sizes)
    sizes.insert(0, total_size)
    if return_quadrants_df:
        return sizes, quadrants
    return sizes


def evaluate_performance_binary_regression(target, prediction, threshold=0, verbose=True, scatter_plot=True):
    """Prints how well the model performs in a binary way, counting the points
    above and below a threshold that are above and below that threshold in the
    target set

    Parameters:
    ----------
    target : pandas.Series
    predictions : pandas.Series
    threshold : numeric, deefault 0
    verbose : bool, default 0

    Returns
    ----------
    accuracy : float
        percentage of times that the model predicted well the return was above or below the
        threshold
    positive_recall : float
        percentage of times that the prediction was above the threshold when the return was
        above the threshold
    negative_recall : float
        percentage of times that the prediction was below the threshold when the return was
        below the threshold
    """

    sizes = count_quadrants(target, prediction, threshold)

    same_sign = sizes[2] + sizes[3]
    accuracy = same_sign / sizes[0] * 100

    positive_percentage_target = (sizes[2] + sizes[4]) / sizes[0] * 100
    positive_recall = (sizes[2] / (sizes[2] + sizes[4])) * 100

    negative_percentage_target = (sizes[1] + sizes[3]) / sizes[0] * 100
    negative_recall = (sizes[3] / (sizes[1] + sizes[3])) * 100

    positive_percentage_prediction = (sizes[1] + sizes[2]) / sizes[0] * 100
    positive_precision = (sizes[2] / (sizes[1] + sizes[2])) * 100

    negative_percentage_prediction = (sizes[3] + sizes[4]) / sizes[0] * 100
    negative_precision = (sizes[3] / (sizes[3] + sizes[4])) * 100



    if verbose:
        print('The model predicts well the quadrant of the return of {:.1f}% of the samples'.format(accuracy))
        print()

        print('In {:.1f}% of the samples, the actual return is above {:.3f}'.format(positive_percentage_target,
                                                                                    threshold))
        print("When the actual return is above {:.3f}, the predicted return is above {:.3f} a {:.1f}% of the time"\
              .format(threshold, threshold, positive_recall))
        print()

        print('In {:.1f}% of the samples, the actual return is below {:.3f}'.format(negative_percentage_target,
                                                                                    threshold))
        print("When the actual return is below {:.3f}, the predicted return is  below {:.3f} a {:.1f}% of the time"\
              .format(threshold, threshold, negative_recall))
        print()

        print('In {:.1f}% of the samples, the predicted return is above {:.3f}'.format(positive_percentage_prediction,
                                                                                       threshold))
        print("When the predicted return is above {:.3f},the actual return is above {:.3f} a {:.1f}% of the time"\
              .format(threshold, threshold, positive_precision))
        print()

        print('In {:.1f}% of the samples, the predicted return is below {:.3f}'.format(negative_percentage_prediction,
                                                                                       threshold))
        print("When the predicted return is below {:.3f}, the actual return is below {:.3f} a {:.1f}% of the time"\
              .format(threshold, threshold, negative_precision))
        print()

    if scatter_plot:
        scatter(target, prediction, threshold)

    return accuracy, positive_recall, negative_recall, positive_precision, negative_precision


def scatter(target, predictions, threshold=0, ideal_prediction_line=True):
    """Plots a scatterplot with real values in the x axis, predictions in the y axis
    and a red horizontal and vertical lines separating the zones in 4 quadrants

    target : pandas.Series
    predictioins : pandas.Series
    threshold : numeric or None
        numeric : draw hline and vline
        None : just the scatter plot
    ideal_prediction_line : bool, default True
        plots a straight line with the best possible fit, where prediction=target"""

    plt.plot(target, predictions, linewidth=0, marker='o', zorder=1);
    if threshold is not None:
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.hlines(threshold, xlim[0], xlim[1], linestyles='dotted', color='red')
        plt.vlines(threshold, ylim[0], ylim[1], linestyles='dotted', color='red')
        plt.xlim(xlim)
        plt.ylim(ylim)
    if ideal_prediction_line:
        plt.plot([target.min(), target.max()], [target.min(), target.max()], label='Ideal prediction')
    plt.legend()
    plt.title('Predicted returns VS actual returns')
    plt.xlabel('Actual return')
    plt.ylabel('Predicted return');


def replaces_nans_ma(series):
    """Substitutes NaN, -inf and inf with the average of all the previous values
    of a pandas.Series"""
    series = series.replace([np.inf, -np.inf], np.nan)
    result = series.fillna(series.rolling(window=len(series), min_periods=0).mean())
    return result


def replace_outliers(series, period=5000, min_period=0, k=5):
    """Remove outliers from an only-positive time-series"""
    log_series = np.log(series)
    std_rolling = log_series.rolling(window=period, min_periods=min_period).std()
    mmed = log_series.rolling(window=period, min_periods=min_period).median()
#     ma = log_series.rolling(window=period, min_periods=min_period).mean()
    threshold = k * std_rolling

    index_values_to_replace = np.abs(log_series - mmed) > threshold
    new_series = series.copy()
    new_series[index_values_to_replace] = np.exp(mmed)
    return new_series

def multiprocess_pandas_dataframe(func, df, num_processes=None, **kwargs):
    ''' Apply a function separately to each column in a dataframe, in parallel.'''

    # If num_processes is not specified, default to minimum(#columns, #machine-cores)
    if num_processes==None:
        num_processes = min(df.shape[1], cpu_count())

    # 'with' context manager takes care of pool.close() and pool.join() for us
    with Pool(num_processes) as pool:
        # we need a sequence of columns to pass pool.map
        seq = [df[column_name] for column_name in df]

        # pool.map returns results as a list
        results_list = pool.map(partial(func, **kwargs), seq)

        # return list of processed columns, concatenated together as a new dataframe
        return pd.concat(results_list, axis=1)

def vel_acc_series(series, periods=[1], geom=True):
    """Compute the geometric and simple velocity and acceleration of series over
    variation_periods. If a column of df is the result of a study over a certain
    period N, its velocity is only calculated over periods in variation_periods that
    are less or equal than 1.2 * N

    If the column has negative values, only simple velocity and acceleration is
    calculated, no geometric.

    vel[i] = (value[i] - value[i - n]) / n
    acc[i] = vel[i] - vel[i] - 1
    geom_vel = value[i] / value[i - n]
    geom_acc = geom_vel[i] / geom_vel[i - 1]
    log(geom_vel) = np.log(geom_vel)
    log(geom_acc) = np.log(geom_acc)


    Parameters
    ----------
    df : pandas.DataFrame
        where the time-series are
    log : bool, default True
        compute log geometric velocity and acceleration or non-log
    periods : list
        list of integers to look back in the past to calculate returns
    geom : bool, default True
        calculate geometric velocity and acceleration or not


    Returns
    ----------
    result : pandas.DataFrame"""

    fragments = re.findall(r'_\d+', series.name)
    if fragments:
        # gets the first integer found in the previous found substrings
        col_period = [int(string) for string in re.findall(r'\d+',
                                                           fragments[0])][0]
        # creates a list with periods minor than 1.2 * the integer from above
        lower = [n for n in periods if n <= 1.2 * col_period]
    else:
        # if no integers are found in the column name, ROC is calculated over all
        # the possible pperiods
        lower = periods


    dfs = []
    for n in lower:
        vel = series.diff(n) / n
        acc = vel.diff()
        vel.name = "VEL_{} ({})".format(str(n), series.name)
        acc.name="ACC_{} ({})".format(str(n), series.name)

        dfs.append(vel)
        dfs.append(acc)


        if geom:
            if not (series <= 0).any():
                returns = series.pct_change(periods=n) + 1
                geom_vel = returns.pow(1 / n)
                geom_vel.name = 'GEOM_VEL_{} ({})'.format(str(n), series.name)
                dfs.append(geom_vel)

                log_geom_vel = np.log(geom_vel)
                log_geom_vel.name = 'log ({})'.format(geom_vel.name)
                dfs.append(log_geom_vel)

                geom_acc = geom_vel.pct_change() + 1
                geom_acc.name = 'GEOM_ACC_{} ({})'.format(str(n), series.name)
                dfs.append(geom_acc)

                log_geom_acc = np.log(geom_acc)
                log_geom_acc.name = 'log ({})'.format(geom_acc.name)
                dfs.append(log_geom_acc)


    return pd.concat(dfs, axis=1)

def compute_vel_acc(df, periods=variation_periods, log=True, geom=True):
    """Compute the geometric and simple velocity and acceleration of cols in df over
    variation_periods. If a column of df is the result of a study over a certain
    period N, its velocity is only calculated over periods in variation_periods that
    are less or equal than 1.2 * N

    If the column has negative values, only simple velocity and acceleration is
    calculated, no geometric.

    vel[i] = (value[i] - value[i - n]) / n
    acc[i] = vel[i] - vel[i] - 1
    geom_vel = value[i] / value[i - n]
    geom_acc = geom_vel[i] / geom_vel[i - 1]
    log(geom_vel) = np.log(geom_vel)
    log(geom_acc) = np.log(geom_acc)


    Parameters
    ----------
    df : pandas.DataFrame
        where the time-series are
    log : bool, default True
        compute log geometric velocity and acceleration or non-log
    periods : list
        list of integers to look back in the past to calculate returns
    geom : bool, default True
        calculate geometric velocity and acceleration or not


    Returns
    ----------
    result : pandas.DataFrame
    """
    lower = {}
    result = pd.DataFrame(index=df.index)
    non_geom_list = []
    geom_list = []
    for col, values in df.iteritems():
        # find all occurrences of the substring (_ + a number) in the columns of df
        fragments = re.findall(r'_\d+', col)
        if fragments:
            # gets the first integer found in the previous found substrings
            col_period = [int(string) for string in re.findall(r'\d+',
                                                               fragments[0])][0]
            # creates a list with periods minor than 1.2 * the integer from above
            lower[col] = [n for n in periods if n <= 1.2 * col_period]
        else:
            # if no integers are found in the column name, ROC is calculated over all
            # the possible pperiods
            lower[col] = periods

        non_geom_temp = compute_function_different_periods(df=df,
                                                      function=ta.VEL_ACC,
                                                      periods=lower[col],
                                                      col=col)
        non_geom_list.append(non_geom_temp)
        if geom:
            if not (values <= 0).any():
                geom_temp = compute_function_different_periods(df=df,
                                                          function=ta.GEOM_VEL_ACC,
                                                          periods=lower[col],
                                                          col=col,
                                                          log=log)
                geom_list.append(geom_temp)

    result = pd.concat(non_geom_list, axis=1)
    if geom:
        geom_df = pd.concat(geom_list, axis=1)
        result = pd.concat([result, geom_df], axis=1)

    return result


def compute_Y(X, periods=[1], log=True, return_X=False, add_one=True):
    """Computes returns to predict from a train dataframe X for n periods
    y(i, n) = X.Close[i + n] / X.Close[i]
    if log, y = log(1 + y)

    Parameters
    ----------

    X : pandas.DataFrame
        train features
    periods : list of integers, default [1]
        periods to generate target data over
    log : bool, default True
        calculates simple or log returns
    return_X : bool, default False
        True :  returns X with the rows that don't have a corresponding Y dropped
        False : returns only Y
    add_one : bool, default True
        if the return is not logarithmic, adds one. This way it's centered
        around one and we can compute relative errors without having divisions
        by zero

    Returns
    ----------
    Y : pandas.DataFrame
        returns to predict later with ML"""

    Y = compute_function_different_periods(df=X,
                                           periods=periods,
                                           function=compute_target,
                                           log=log)

    if not log and add_one:
       Y += 1
    if return_X:
        X = X.join(Y).dropna()
        Y = X[Y.columns]
        X = X[X.columns.difference(Y.columns)]
        return X, Y
    else:
        return Y



def technical_analysis(df, periods=ta_periods, macd_periods=periods_MACD):
    """Performs several technical analysis on the OHLCV data"""

    original_join_state = SETTINGS.join
    SETTINGS.join = False

    if 'Volume' not in df.columns and cols_in_df(df, ['Vol']) != []:
        df['Volume'] = df['Volume (BTC)'] # some TA functions need a 'Volume' column

    if cols_in_df(df, ['Vol']) != []:
        result = pd.concat([compute_function_different_periods(df, periods, ta.ATR),
                            compute_function_different_periods(df, periods, pta.BBANDS),
                            compute_function_different_periods(df, periods, pta.STO),
                            compute_function_different_periods(df, periods, pta.TRIX),
                            # Vortex is a FUCKIN SHIT that gives randomly high values. Fuck it
                            # compute_function_different_periods(df, [period for period in periods if period > 6], pta.Vortex),
                            compute_function_different_periods(df, periods, pta.RSI),
                            # compute_function_different_periods(df, periods, pta.ACCDIST),
                            compute_function_different_periods(df, periods, pta.MFI),
                            compute_function_different_periods(df, periods, pta.OBV),
                            compute_function_different_periods(df, periods, pta.FORCE),
                            # compute_function_different_periods(df, periods, pta.EOM),
                            compute_function_different_periods(df, periods, pta.CCI),
                            compute_function_different_periods(df, periods, pta.COPP),
                            compute_function_different_periods(df, periods, pta.KELCH),
                            compute_function_different_periods(df, periods, pta.STDDEV),
                            compute_function_different_periods(df, periods, pta.MA),
                            compute_function_different_periods(df, periods, ta.MMed),
                            compute_function_different_periods(df, periods, pta.EMA),
                            # compute_function_different_periods(df, periods, pta.MOM),
                            # compute_function_different_periods(df,periods,  pta.ROC),
                            # compute_function_different_periods(df, ROC, log=True),
                            # pta.MACD(df, 10, 30),

                            compute_MACD_different_periods(df, periods=macd_periods)
                            # pta.PPSR(df)
                            ], axis=1)

    else:
        result = pd.concat([compute_function_different_periods(df, periods, ta.ATR),
                            compute_function_different_periods(df, periods, pta.BBANDS),
                            compute_function_different_periods(df, periods, pta.STO),
                            compute_function_different_periods(df, periods, pta.TRIX),
                            compute_function_different_periods(df, periods, pta.RSI),
                            compute_function_different_periods(df, periods, pta.CCI),
                            compute_function_different_periods(df, periods, pta.COPP),
                            compute_function_different_periods(df, periods, pta.KELCH),
                            compute_function_different_periods(df, periods, pta.STDDEV),
                            compute_function_different_periods(df, periods, pta.MA),
                            compute_function_different_periods(df, periods, ta.MMed),
                            compute_function_different_periods(df, periods, pta.EMA),
                            compute_MACD_different_periods(df, periods=macd_periods)
                            ], axis=1)


    # result = result.fillna(method='pad')
    SETTINGS.join = original_join_state
    return out(SETTINGS, df, result)

def standarize(train_data, test_data, dropna=True):
    """Standarizes train and test features by removing the mean and scaling to unit variance

    Calendar data should be calculated after this process. If not, when we standarize data
    that has a timespan minor than a full year, some columns will have only zeros and that
    produces nans in that col when dividing by std=0"""
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std
    if dropna:
        train_data = train_data.dropna()
        test_data = test_data.dropna()

    return train_data, test_data


def compute_calendar_functions(df, join=True):
    """Creates columns to account for what month, week, day of the week and day
    of the month we are"""

    # Converts the index to a Datetime Index
    df.index = pd.to_datetime(df.index)
    result = pd.DataFrame(index=df.index)

    # Integer col
    result['Day of month'] = df.index.day

    # Integer col
    result['Day of the week'] = df.index.dayofweek

    # Create 7 boolean columns, one for each day of the week
    week_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i in range(7):
        result[week_days[i]] = (df.index.weekday == i).astype(float)

    # Create 12 boolean columns, one for each month
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
              'Oct', 'Nov', 'Dec']
    for i in range(12):
        result[months[i]] = (df.index.month == i + 1).astype(float)

    # Create 24 boolean colums, one for each hour
    hours = ['Hour ' + str(n) for n in range(24)]
    for i in range(24):
        result[hours[i]] = (df.index.hour == i).astype(float)

    # Create 52 boolean colums, one for each week
    weeks = ['Week ' + str(n + 1) for n in range(53)]
    for i in range(53):
        result[weeks[i]] = (df.index.weekofyear == i + 1).astype(float)

    # Create 31 boolean columns, one for each day of the month
    month_days = ['Day ' + str(n) for n in range(31)]
    for i in range(31):
        result[month_days[i]] = (df.index.day == i + 1).astype(float)

    if join:
        result = df.join(result)
    return result



def compute_variations(df, log=True, periods=variation_periods, verbose=False):
    """Compute the ROC (same thing as returns) or the simple variation over
    variation_periods. If a column of df is the result of a study over a certain
    period N, its ROC is only calculated over periods in variation_periods that
    are less or equal than 1.2 * N

    If the column has negative values, the simple variation is calculated because
    if we calculate the ROC near 0 it reaches very high values (as we're dividing
    by ~0) and the resulting data is not useful.

    Simple variation (n) = value(i) - value(i - n)
    Simple ROC(period=n) = value(i) / value(i - n) - 1
    Log ROC(period=n) = log(simple ROC)


    Parameters
    ----------
    df : pandas.DataFrame
        where the time-series are
    log : bool, default True
        compute simple or log returns
    periods : list
        list of integers to look back in the past to calculate returns


    Returns
    ----------
    result : pandas.DataFrame
    """

    original_join_state = SETTINGS.join
    SETTINGS.join = False

    # Creates a dictionary to store over which periods we're gonna calculate
    # past returns for each of the columns of the DataFrame
    lower = {}
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        # find all occurrences of the substring (_ + a number) in the columns of df
        fragments = re.findall(r'_\d+', col)
        if fragments:
            # gets the first integer found in the previous found substrings
            col_period = [int(string) for string in re.findall(r'\d+',
                                                               fragments[0])][0]
            # creates a list with periods minor than 1.2 * the integer from above
            lower[col] = [n for n in periods if n <= 1.2 * col_period]
        else:
            # if no integers are found in the column name, ROC is calculated over all
            # the possible pperiods
            lower[col] = periods

        # If the column has negative values, we calculate simple variation with ta.MOM
        if False:
            pass
            # (df[col]<= 0).any():
            # method='MOM'
            # MOMs = compute_function_different_periods(df=df,
            #                                           function=ta.MOM,
            #                                           periods=lower[col],
            #                                           col=col)
            # result = result.join(MOMs)

        # Otherwise, calculate ROC
        else:
            method='ROC'
            ROCs = compute_function_different_periods(df=df,
                                                  function=ROC,
                                                  periods=lower[col],
                                                  col=col,
                                                  log=log)
            result = result.join(ROCs)
        if verbose:
            print(col, method)

    SETTINGS.join = original_join_state
    return out(SETTINGS, df, result)


def compute_target(df, log=True, col='Close', n=1):
    """Computes the returns for n periods ahead for the specified column

    Parameters
    ------------
    df : pandas.DataFrame
        containing data where col is
    col : str
        name of the col which returns we want to predict


    Returns
    ----------
    result : returns that happened n perios into the future"""

    result = ROC(df=df, log=log, col=col, n=n)
    if log:
        result = np.log(1 + result)
    result = result.shift(-n)
    result.name = 'TARGET_' + str(n)

    return out(SETTINGS, df, result)


def dynamic_standarization(df, n=365):
    """Standardize features by removing the mean and scaling to unit variance,
    using rolling mean and standard deviation for n periods

    Parameters
    ----------
    df : pandas.DataFrame
        data to be dynamically standarized
    n : period of the rolling window

    Returns
    ----------
    result : pandas.DataFrame
        standarized data
    """
    mov_avg = df.rolling(window=n).mean()
    mov_std = df.rolling(window=n).std()
    result = (df - mov_avg).div(mov_std)

    # I do this fillna(df) thing because of when calculating std of some blockchain
    # data being 0 produces way too many NaNs and leaves us with roughly 2 years of data
    result = result.fillna(df)
    result = result.iloc[n-1:]

    return out(SETTINGS, df, result)


def walk_forward_split(data, in_sample_periods=None, out_of_sample_periods=None, print_indexes=False):
    """Creates train and test indexes to perform a walk-forward analysis

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series, array-like
        data we want to use for training and testing
    in_sample_periods : int
        number of periods we want to use for testing
    out_of_sample_periods : int
        number of periods we want to use for testing
    print_indexes : bool, default False
        wether to print or not the indexes used in each step for training and testing

    Returns
    ----------
    indexes : list
        list of tuples with of the form (train_index, test_index)
    """
    try:
        index = data.index # if data is a pandas-like variable
    except:
        index = np.arange(len(data)) # is data is an array
    if not out_of_sample_periods:
        out_of_sample_periods = int(round(len(index) / 10 ))
    if not in_sample_periods:
        in_sample_periods = out_of_sample_periods * 5
    if len(index) < in_sample_periods + out_of_sample_periods + 1:
        raise(ValueError('Data not big enough for the in-sample and out-of sample specified periods'))
    total_index_size = in_sample_periods + out_of_sample_periods
    indexes = []
    n_splits = len(index) // out_of_sample_periods + 1
    for i in range(n_splits):
        train_index = index[i * out_of_sample_periods :
                            i * out_of_sample_periods + in_sample_periods]
        test_index = index[i * out_of_sample_periods + in_sample_periods :
                           i * out_of_sample_periods + in_sample_periods + out_of_sample_periods]
        indexes.append((train_index, test_index))
        if len(test_index) < 1:
            break
        if print_indexes:
            try:
                print("TRAIN:", [h for h in train_index.hour],
                      "TEST:", [h for h in test_index.hour])
            except:
                print("TRAIN:", train_index, "TEST:", test_index)
    return indexes


def cols_in_df(df, partial_col_names, not_present=None):
    """Returns a list containing the columns in df that contain any
    of the strings in selected_cols in it

    Parameters
    ----------
    df : pandas.DataFrame
        pandas.DataFrame where the data is
    cols_partial_names : list of strings
        substrings of strings which are columns in the main dataframe
        and we want to select
    not_present : list of strings
        substrings of strings from columns we don't want to select

    Returns
    ----------
    result : list of the columns of the main DataFrame

    """

    present = set([col for col in df.columns
                       for part in partial_col_names
                            if part in col])
    if not_present:
        to_exclude = set([col for col in present
                              for part in not_present
                                  if part in col])
        result = list(present.difference(to_exclude))
    else:
        result = list(present)
    return result

def only_positive_values(df):
    """Returns a subset pandas.DafaFrame of df that only has positive values"""


    only_positive_cols_bool = (df <= 0).any()
    only_positive_cols = only_positive_cols_bool[~only_positive_cols_bool].index
    positive_df = df[only_positive_cols]

    return positive_df

def log_transform(df, cols=None):
    """Returns the log of the given columns of the given Dataframe.
    Only calculates the logarithms of the colums that have positive values

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame where the data is
    cols : list
        list of columns of the dataframe

    Returns
    ----------
    result : pandas.DataFrame
        np.log of each of the columns
        """

    if not cols:
        small_df = only_positive_values(df)
    else:
        cols = cols_in_df(df, partial_col_names=columns_to_transform_log,
                          not_present=['MACD'])
        small_df = df[cols]
        small_df = only_positive_values(df[cols])
    result = np.log(small_df)
    result.columns = ['log ({})'.format(col) for col in small_df]
    # if replace_infs:
    #     # result = result.replace(-np.inf, np.nan)
    #     for col in result:
    #         result[col] = result[col].fillna(result[col].min())
    return out(SETTINGS, df, result)


def compute_function_different_periods(df, periods=periods, function=None,
                                       **kwargs):
    """Compute the same function from pandas_talib altering its parameter n

    Parameters
    ----------
        df : df
            pandas.Dataframe where the time-series is
        function : string
            Name of the oscillator we want to calculate
        periods : array-like
            List or iterable containing the different values (typically periods)
            to calculate the oscillator time-series to each of them

    Returns:
    ----------
        result : pandas.Dataframe with the different time-series for each of
            the periods appended
    """

    index = df.index
    df_resetted_index = df.reset_index()

    original_join_state = SETTINGS.join
    original_join_state_pta = pta.SETTINGS.join
    pta.SETTINGS.join = False
    SETTINGS.join = False

    # result = []
    for n in periods:
        serie = function(df=df_resetted_index, n=n, **kwargs)
        try:
            result = result.join(serie)
        except NameError:
            result = pd.DataFrame(serie)
        # series.name = series.name + '_{}'.format(str(n))
    # result = pd.concat(result, axis=1)
    result.index = index

    SETTINGS.join = original_join_state
    pta.SETTINGS.join = original_join_state_pta

    return result


def compute_MACD_different_periods(df, periods, **kwargs):
    """Computes the MACD function over a list of tuples of different periods

    Parameters
    ----------
        df : df
            pandas.Dataframe where the time-series is
        periods : list
            List of tuples with the form [(n_slow_1, n_fast_1),
                                          (n_slow_2, n_fast_),...]
                containing the slow and fast MACD periods

    Returns:
    ----------
        result : pandas.Dataframe with the different time-series for each of
            the periods appended"""

    index = df.index
    df_resetted_index = df.reset_index()

    original_join_state = SETTINGS.join
    original_join_state_pta = pta.SETTINGS.join
    pta.SETTINGS.join = False
    SETTINGS.join = False

    for tup in periods:
        serie = pta.MACD(df=df_resetted_index, n_fast=tup[0],
                         n_slow=tup[1], **kwargs)
        try:
            result = result.join(serie)
        except NameError:
            result = pd.DataFrame(serie)
    result.index = index

    SETTINGS.join = original_join_state
    pta.SETTINGS.join = original_join_state_pta

    return out(SETTINGS, df, result)



def ROC(df, col='Close', log=True, n=1):
    """Receives a pandas Dataframe and returns the simple or log returns
    from n days back from column values as a pandas.Series object, or appends
    it to the df

    Parameters:
    ----------
        df : pandas.Dataframe or pandas.Series
        col : string, default 'Close'
             Name of the column whose returns we want to calculate
        log : Bool, default false
             False : calculates the simple returns, as price[i] / price[i-n] - 1
             True : calculates the log returns, as log (price[i] / price[i-n])
        n : int, default 1
            number of periods to look back to calculate the return

    Returns:
    ----------
        result : pandas.Series or pandas.DataFrame

    """

    values = df[col]
    result =  values / values.shift(n) - 1
    result.name='ROC_{} ({})'.format(str(n), result.name)
    if log:
        result = np.log(result + 1)
        result.name = 'log ({})'.format(result.name)

    return out(SETTINGS, df, result)

    # if type(df) == pd.DataFrame:
    #     values = df[col] # selects column col from the Dataframe
    # else:compute_ROCs(result, log=True, periods=variation_periods)
    #     values = df # yet it'd be a pd.Series
    #
    # result = pd.Series(index=values.index)
    # result = values / values.shift(n) - 1
    # result[0:n] = 0
    # result = pd.Series(data=result,
    #                    name='ROC_{} ({})'.format(str(n), values.name))
    #
    # if log:
    #     result = np.log(1 + result)
    #     result.name = 'log ({})'.format(result.name)
    #
    # if type(df) == pd.DataFrame:
    #     return out(SETTINGS, df, result)
    # return result


def nondimensionalize(df, cols=[], series=None, inverse=False):
    """Nondimensionalizes the inputed Dataframe, dividing all of its values by
    the valuesfrom column col of that Dataframe, and returns the
    nondimensionalized df without the col

    Parameters
    ----------
        df : pandas.Dataframe
            DataFrame to be nondimensionalized
        cols : List, default None
            columns of the dataframe di divide by series
        series : pandas.Series, default None
            time-series to nondimensionalize the DataFrame with respect to it
        inverse : bool, default False
            False : divides the DataFrame columns by the indicated col
            True : divides the indicated col by the other columns of the
                Dataframe

    Returns
    ----------
        result : pandas.DataFrame
            Nondimensionalized DataFrame
    """

    if not cols:
        temp_df = df
    else:
        temp_df = df[cols]

    if type(series) is pd.Series:
        pass
    elif type(series) is str:
        series = df[series]
    elif series is None:
        series = df['Close']


    result = temp_df.divide(series, axis=0)

    if inverse:
        result = 1 / result
        result.columns = [series.name + ' / ' + name for name in temp_df.columns]
    else:
        result.columns = [name + ' / ' + series.name for name in temp_df.columns]

    return out(SETTINGS, df, result)



def compute_mayer_multiples(df, price='Close', averages = ['MA', 'SMA']):
    """Calculates Mayer Multiples, defined as Close / SMA(n), for the ns present
    in the df

    Parameters:
    ----------
    df : pandas.DataFrame
        containing columns corresponding to Close and several periods_MACD_short
    averages : List
        compute the MM with MA, SMA, both...
    price : str, default 'Close'
        name of the col where the price to divide by the MAs is

    Returns
    ----------
    result : pandas.DataFrame
        containing the Mayer Multiples and the log(Mayer Multiples)
    """
    original_join_state = SETTINGS.join
    SETTINGS.join = False

    not_present = ['log', 'MACD', 'EMA', 'MA']
    not_present = [el for el in not_present if el not in averages]

    cols_df = cols_in_df(df, partial_col_names=averages, not_present=not_present)
    mayer_multiples_df = nondimensionalize(df, cols_df, price, inverse=True)
    log_mayer_multiples_df = log_transform(mayer_multiples_df)
    result = pd.concat([mayer_multiples_df, log_mayer_multiples_df], axis=1)

    SETTINGS.join = original_join_state
    return out(SETTINGS, df, result)


def compute_ratios_ta(df, price='Close', cols=columns_to_divide):
    """Divides specified columns by price, to nondimensionalize themew

    Parameters:
    ----------
    df : pandas.DataFrame
        containing columns corresponding to Close and several periods_MACD_short
    cols : List of strings
        names of the columns to be divided by price
    price : str, default 'Close'
        name of the col where the price to divide by the MAs is

    Returns
    ----------
    result : pandas.DataFrame
        containing the ratios and the log(ratios)
    """
    original_join_state = SETTINGS.join
    SETTINGS.join = False

    not_present = ['log']
    not_present = [el for el in not_present if el not in cols]

    cols_df = cols_in_df(df, partial_col_names=cols, not_present=not_present)
    ratios_df = nondimensionalize(df, cols_df, price, inverse=False)
    log_ratios_df = log_transform(ratios_df)
    result = pd.concat([ratios_df, log_ratios_df], axis=1)

    SETTINGS.join = original_join_state
    return out(SETTINGS, df, result)


def rolling_max(series, n=None):
    """Returns the rolling maximum value of the series

    Parameters
    ----------
    series : pandas.Series
        tiem-series to calculate the rolling max
    n : int, default None
        None : calculates the max value of series to date
        int : calculates the rolling max value of the series,
            with a rolling window of n

    Returns
    ----------
    result : pandas.Series
        rolling max"""

    if n:
        result = series.rolling(window=n, min_periods=0).max()
        result.name = 'rolling_max_' + str(n)
    else:
        result = series.rolling(window=len(series), min_periods=0).max()
        result.name = 'max_to_date'

    return result

def drawdown(series, n=None):
    """Calculates Drawdown, as Price / rolling_max(price). From 0 and 1, at ATH is 1"""

    roll_max = rolling_max(series, n)
    drawdown = series.divide(roll_max)

    result = drawdown
    if not n:
        result.name = 'Drawdown'
    else:
        result.name = 'Drawdown_{}'.format(str(n))

    return result

def inverse(series):
    """Returns 1 / series"""

    result = 1 / series
    result.name = 'inv ({})'.format(series.name)

    return result

def substract_1(series):
    """Returns 1 - series"""

    result = series - 1
    result.name = series.name + ' -1'

    return result

def substract_from_1(series):
    """Returns 1 - series"""

    result = 1 - series
    result.name = '1 - ' + series.name

    return result

def compute_derived_drawdown_functions(df, col='Close', n=None):
    """
    Calculates:
    dd : drawdown, calculated as Price / rolling_max(price). From 0 and 1, at ATH is 1
    dd_inv : 1 / drawdown. From 1 to inf, at ATH is 1
    subs_1_dd_inv : 1 - (1 / drawdown). From 0 to inf, at ATH is 1
    log_dd_neg : - np.log(dd). From -inf to 0, at ATH is 0

    ATH is all-time high. Max price seen to date.


    Parameters
    ----------
    series : pandas.DataFrame
        where the data is
    col : str
        name of the col of the df containing price data
    col : str
        col of df where the data is
    n : int, default None
        None : calculates the max value of series to date
        int : calculates the rolling max value of the series,
            with a rolling window of n

    Returns
    ----------
    result : pandas.Dataframe
        all 4 series combined, or the full df

    """

    price = df[col]

    dd = drawdown(price, n=n)
    subs_from_1_dd = substract_from_1(dd)
    log_dd_neg = - np.log(dd)
    log_dd_neg.name = '- log ({})'.format(dd.name)
    dd_inv = inverse(dd)
    subs_1_dd_inv = substract_1(dd_inv)

    result = pd.concat([dd, subs_from_1_dd, log_dd_neg, dd_inv, subs_1_dd_inv], axis=1)

    return result



def trade_vol_VS_tx_vol(df):
    """Data showing the relationship between BTC traded volume and transatcions
    volume

    Quandl data is BAD. I build my own.
    """

    volume_cryptocompare = df['Volume (BTC)']
    volume_tx = df['Tx Volume (BTC)']
    result = volume_cryptocompare.div(volume_tx).fillna(0)
    result.name = 'Trade Vol / Tx Vol'
    return out(SETTINGS, df, result)


def miners_revenue_VS_tx_volume(df):
    """Daily data showing miners revenue as as percentage
    of the transaction volume.

    Quandl data is has values way too high for some points, around 1e7.
    Seems useless. I build my own data better.
    URL: https://www.quandl.com/data/BCHAIN/CPTRV
    """
    miners_revenue_USD = df['Miners Revenue (USD)']
    tx_vol_USD = df['Tx Volume (USD)']
    revenue_VS_tx_vol = miners_revenue_USD.div(tx_vol_USD)
    result = revenue_VS_tx_vol
    result.name = 'Miners Revenue / Tx Volume'
    return out(SETTINGS, df, result)


def avg_tx_fees_USD(df):
    """Average transaction fees in USD"""
    result = df['Tx fees (USD)'].div(df['Txs'])
    result.name = 'Avg Tx Fees (USD)'
    return out(SETTINGS, df, result)


def avg_tx_fees_BTC(df):
    """Average transaction fees in BTC"""
    result = df['Tx fees (BTC)'].div(df['Txs'])
    result.name = 'Avg Tx Fees (BTC)'
    return out(SETTINGS, df, result)


def block_reward_USD(df):
    """USD value of the mining rewards per block

    Computed as the Total Mining Revenue - Transaction Fees"""

    miners_revenue_USD = df['Miners Revenue (USD)']
    tx_fees_USD = df['Tx fees (USD)']
    result = miners_revenue_USD - tx_fees_USD
    result.name = 'Block Reward (USD)'
    return out(SETTINGS, df, result)

def tx_fees_VS_miners_revenue(df):
    """Proportion of the miners revenue that corresponds to fees"""

    miners_revenue_USD = df['Miners Revenue (USD)']
    tx_fees_USD = df['Tx fees (USD)']
    result = tx_fees_USD.div(miners_revenue_USD)
    result.name = 'Tx Fees / Miners Revenue'
    return out(SETTINGS, df, result)


def avg_tx_value_USD(df):
    """Average transaction value in USD"""

    tx_vol_USD = df['Tx Volume (USD)']
    daily_txs = df['Txs']
    result = tx_vol_USD.div(daily_txs)
    result.name = 'Avg Tx Value (USD)'
    return out(SETTINGS, df, result)


def avg_tx_value_BTC(df):
    """Average transaction value in BTC"""

    tx_vol_BTC = df['Tx Volume (BTC)']
    daily_txs = df['Txs']
    result = tx_vol_BTC.div(daily_txs)
    result.name = 'Avg Tx Value (BTC)'
    return out(SETTINGS, df, result)


def fee_VS_tx_value(df):
    """Proportion of an average transaction that correspond to its fees"""

    total_fees = df['Tx fees (USD)']
    tx_vol_USD = df['Tx Volume (USD)']
    result = total_fees.div(tx_vol_USD)
    result.name = 'Tx Fees / Tx Volume'
    return out(SETTINGS, df, result)



def velocity(df, n=90, join=True):
        """Bitcoin's velocity is calculated by dividing the 90 day estimated USD transaction
        volume by the 90 day average USD market cap. (This is the equivalent to the $BTC
        circulated divided by the Bitcoin money supply.)

        http://charts.woobull.com/bitcoin-velocity/

        Parameters
        ----------
            df : pandas.Dataframe, default None
                None : fetches all data from the Internet
                If provided, creates tha velocity data from the columns of the df
            n : integer, default 90
                period over which the moving averages and sums are calculated
            join : boolean, default False
                True : Appends the original df + a column containing velocity data,
                    returns df
                False : Returns pandas.Series

        Returns
        ----------
            result : pandas.Series or pandas.Dataframe
                Bitcoin Velocity
        """

        tx_vol = df['Tx Volume (USD)'].rolling(window=n).sum()
        mkt_cap = df['Market Cap (USD)'].rolling(window=n).mean()

        result = tx_vol.div(mkt_cap)
        result.name = 'Velocity_' + str(n)

        return out(SETTINGS, df, result)


def NVT_ratio(df, n=30, join=True):
    """Bitcoin's NVT is calculated by Network Value (Market Cap) divided by the USD volume
    transmitted through the blockchain. The USD value transmitted is an estimation by
    Blockchain.info

    http://charts.woobull.com/bitcoin-nvt-ratio/

    Parameters
    ----------
        df : pandas.Dataframe, default None
            None : fetches all data from the Internet
            If provided, creates tha velocity data from the columns of the df
        n : integer, default 30
            period over which the moving average is calculated, to smooth the output
        join : boolean, default False
            True : Appends the original df + a column containing velocity data,
                returns df
            False : Returns pandas.Series

    Returns
    ----------
        result : pandas.Series or pandas.DataFrame
            Network Value to Transactions Ratio
    """

    tx_vol = df['Tx Volume (USD)']
    mkt_cap = df['Market Cap (USD)']

    result = mkt_cap.div(tx_vol).rolling(window=n).mean()
    result.name = 'NVT Ratio_' + str(n)

    return out(SETTINGS, df, result)
