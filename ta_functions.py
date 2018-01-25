import pandas as pd
import numpy as np
import pandas_talib as pta

SETTINGS = pta.SETTINGS

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
        return df.join(result)
    return result.squeeze()

def ATR(df, n):
    """
    Average True Range
    """
    i = 0
    TR_l = [0]
    while i < len(df) - 1:  # df.index[-1]:
    # for i, idx in enumerate(df.index)
        # TR=max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR = max((df['High'].iloc[i + 1] - df['Low'].iloc[i + 1]),
                 df['Close'].iloc[i] - min(df['Low'].iloc[i + 1],
                                           df['High'].iloc[i + 1]))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    result = pd.Series(pd.ewma(TR_s, span=n, min_periods=n), name='ATR_' + str(n))
    return out(SETTINGS, df, result)


def MMed(df, n, price='Close'):
    """
    Rolling Median
    """
    name='Med_{n}'.format(n=n)
    result = pd.Series(df[price].rolling(window=n).median(), name=name)
    return out(SETTINGS, df, result)


def MOM(df, n, col='Close'):
    """
    Momentum
    """
    result = pd.Series(df[col].diff(n),
                       name='Momentum_' + str(n) + ' ({})'.format(col))
    return out(SETTINGS, df, result)


def VEL_ACC(df, col=None, n=1, include_acceleration=True):
    """Velocity, calculated as the variation from n periods back divided by n
    Acceleration, calculated as the variation of velocity one from 1 period back

    Parameters
    ----------
    df : pd.Series or pd.DataFrame
        containing the time-series
    col : string
        if df is a pd.DataFrame, name of the column whose velocity and acceleration
        we want to calculate
    n : int, default 1
        number of periods back we want to use to compute velocity
    include_acceleration : bool, default True
        wether to include acceleration or not

    Returns
    ----------
    result : pandas.DataFrame or pandas.Series
        containint both velocity and acceleration or only velocity"""
    if not col:
        series = df
    else:
        series = df[col]
    vel = pd.Series(series.diff(n) / n,
                    name="VEL_{} ({})".format(str(n), series.name))
    if include_acceleration:
        acc = pd.Series(vel.diff(),
                        name="ACC_{} ({})".format(str(n), series.name))
        result = pd.concat([vel, acc], axis=1)
        return result
    else:
        return vel


def GEOM_VEL_ACC(df, col=None, n=1, include_acceleration=True, log=True):
    """Calculates geometric velocity and acceleration

    GEOM_VEL : geometric mean of the returns of the last n values of a time-series
    GEOM_ACC : GEOM_VEL[i] / GEOM_VEL[i-1]

    Parameters
    ----------
    df : pd.Series or pd.DataFrame
        containing the time-series
    col : string
        if df is a pd.DataFrame, name of the column whose velocity and acceleration
        we want to calculate
    n : int, default 1
        number of periods back we want to use to compute velocity
    include_acceleration : bool, default True
        wether to include acceleration or not

    Returns
    ----------
    result : pandas.DataFrame or pandas.Series
        containint both velocity and acceleration or only velocity
    """

    if not col:
        series = df
    else:
        series = df[col]
    returns = series.pct_change(periods=n) + 1
    geom_vel = returns.pow(1 / n)
    geom_vel.name = 'GEOM_VEL_{} ({})'.format(str(n), series.name)
    log_geom_vel = np.log(geom_vel)
    log_geom_vel.name = 'log ({})'.format(geom_vel.name)
    result = pd.DataFrame(log_geom_vel)
    if include_acceleration:
        acc = geom_vel.pct_change() + 1
        acc.name = 'GEOM_ACC_{} ({})'.format(str(n), series.name)
        log_acc = np.log(acc)
        log_acc.name = 'log ({})'.format(acc.name)
        result = result.join(log_acc)

    if not log:
        result = pd.concat([geom_vel, acc], axis=1)
    return result
