import datetime
import requests
import pandas as pd
import quandl
quandl.ApiConfig.api_key = "ko-Mc-BiaTFCKQKo4WPx"

functions =   ['df_price_OHLCV_cryptocompare',
               'tx_fees_BTC',
               'tx_fees_USD',
               'revenue_per_tx_USD',
               'miners_revenue_USD',
               'avg_block_size',
               'blockchain_size',
               'hashrate',
               'exchange_trade_vol',
               'txs_per_block',
               'blockchain_wallet_users',
               'unique_bitcoin_adresses_used_per_day',
               'daily_txs_excluding_from_popular_adressses',
               'daily_txs',
               'total_txs',
               'estimated_tx_volume_BTC',
               'estimated_tx_volume_USD',
               'output_volume_USD',
               # 'network_deficit_USD',
               'total_BTC_mined',
               'mkt_cap_USD',
               'mempool_tx_count',
               'mempool_size_growth_bytes',
               'mempool_size_bytes',
               'UTXO_count',
               'difficulty'
              ]


def blockchain_basic_data():
    """Returns a blockchain contaning data from the Bitcoin blockchain
    fetched from quandl and blockchain.info"""

    df = pd.concat([eval(fun + '()') for fun in functions[1:]], axis=1)
    return df



def full_BTCUSD_basic_data(fillna=True):
    """Creates a pandas.DataFrame with OHLCV and basic data from
    blockchain.info"""

    df_price = df_price_OHLCV_cryptocompare('day')
    df_blockchain = blockchain_basic_data()
    result = df_price.join(df_blockchain)
    if fillna:
        result = result.fillna(method='pad').fillna(method='bfill')
    return result


def df_price_OHLCV_cryptocompare(timeframe='day', symbol='BTC', comparison_symbol='USD', limit=0,
                        aggregate=1, exchange='CCCAGG', print_url=False):
    """Returns a pandas.Dataframe containing OHLC daily data for the specified
    pair of symbols

    Parameters
    ----------
        timeframe : {'day', 'hour', 'minute'}, default 'day'
            'day' : get daily data
            'hour' : get data for each hour
            'minute' : get data for each minute
        symbol : string
            currency we buy, default BTC
        comparison_symbol : string
            currency we sell, default USD
        limit : int
            0 : returns all data available
            > 0 : returns last n rows
        exchange : string, optional
            exchange which we want to get the data from
        print_url : bool, default False
            prints the url that makes the call to the API

    Returns
    ----------
        df : pandas.DataFrame
            DataFrame containing OHLCV historical data
                OHLC : Open, High, Low, Close
                Volume : Volume in the currency being bought
                Volume (...) : Volume in the currency being sold
    """

    url = 'https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym={}&limit={}&aggregate={}&e={}'\
            .format(timeframe, symbol.upper(), comparison_symbol.upper(), limit-1, aggregate, exchange)
    if not limit:
        url += '&allData=True'
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['Date'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    df.index = pd.to_datetime(df.Date)
    if timeframe == 'day':
        df.index = df.index.normalize() # so that the index has only date, not hour. Easier to merge wit other data
    df = df.drop(['Date', 'time'], axis=1)
    df.columns = ['Close', 'High', 'Low', 'Open',
                  'Volume',
                  # 'Volume']
                  'Volume ({})'.format(comparison_symbol)]
    if print_url:
        print(url)
    return df


def tx_fees_BTC():
    """Data showing the total BTC value of transaction fees miners earn per day.

    URL: https://www.quandl.com/data/BCHAIN/TRFEE"""
    result = quandl.get("BCHAIN/TRFEE")
    result.columns = ['Tx fees (BTC)']
    return result.squeeze()


def tx_fees_USD():
    """Data showing the total USD value of transaction fees miners earn per day.

    URL: https://www.quandl.com/data/BCHAIN/TRFUS"""
    result = quandl.get("BCHAIN/TRFUS")
    result.columns = ['Tx fees (USD)']
    return result.squeeze()


def revenue_per_tx_USD():
    """Daily data showing miners revenue divided by the number of transactions.

    URL: https://www.quandl.com/data/BCHAIN/CPTRA"""
    result = quandl.get("BCHAIN/CPTRA")
    result.columns = ['Revenue per tx (USD)']
    return result.squeeze()


def miners_revenue_USD():
    """Daily data showing (number of bitcoins mined per day + transaction fees) *
    market price.

    URL: https://www.quandl.com/data/BCHAIN/MIREV"""
    result = quandl.get("BCHAIN/MIREV")
    result.columns = ['Miners Revenue (USD)']
    return result.squeeze()


def avg_block_size():
    """Daily average block size in MB.

    URL: https://www.quandl.com/data/BCHAIN/AVBLS"""
    result = quandl.get("BCHAIN/AVBLS")
    result.columns = ['Avg Block Size (MB)']
    return result.squeeze()


def blockchain_size():
    """Total size of block headers and transactions in GB. Not including
    database indexes

    URL: https://www.quandl.com/data/BCHAIN/BLCHS"""
    result = quandl.get("BCHAIN/BLCHS") / 1000
    result.columns = ['Blockchain Size (GB)']
    return result.squeeze()


def hashrate():
    """Estimated number of giga hashes per second (billions of hashes
    per second) the bitcoin network is performing.

    URL: https://www.quandl.com/data/BCHAIN/HRATE"""
    result = quandl.get("BCHAIN/HRATE")
    result.columns = ['Hash Rate (GH)']
    return result.squeeze()


def exchange_trade_vol():
    """Daily data showing the USD trade volume from the top exchanges.

    URL: https://www.quandl.com/data/BCHAIN/TRVOU"""
    result = quandl.get("BCHAIN/TRVOU")
    result.columns = ['Exchange Trade Volume (USD)']
    return result.squeeze()


def txs_per_block():
    """Daily average number of transactions per block.

    URL: https://www.quandl.com/data/BCHAIN/NTRBL"""
    result = quandl.get("BCHAIN/NTRBL")
    result.columns = ['Avg TXs per block']
    return result.squeeze()


def blockchain_wallet_users():
    """Number of wallets hosts using  My Wallet Service from blockchain.info

    URL: https://www.quandl.com/data/BCHAIN/MWNUS"""
    result = quandl.get("BCHAIN/MWNUS")
    result.columns = ['Blockchain.info wallet users']
    result = result.interpolate(method='pchip')
    result = result.fillna(method='pad')
    return result.squeeze()


def unique_bitcoin_adresses_used_per_day():
    """Number of unique bitcoin addresses used per day.

    URL: https://www.quandl.com/data/BCHAIN/NADDU"""
    result = quandl.get("BCHAIN/NADDU")
    result.columns = ['Unique Adresses Used per day']
    return result.squeeze()


def daily_txs_excluding_from_popular_adressses():
    """Data showing the total number of unique bitcoin transactions per day
    excluding those which involve any of the top 100 most popular addresses
    popular addresses.

    URL: https://www.quandl.com/data/BCHAIN/NTREP"""
    result = quandl.get("BCHAIN/NTREP")
    result.columns = ['Txs (exl popular addresses)']
    return result.squeeze()


def daily_txs():
    """Total number of unique bitcoin transactions per day..

    URL: https://www.quandl.com/data/BCHAIN/NTRAN"""
    result = quandl.get("BCHAIN/NTRAN")
    result.columns = ['Txs']
    return result.squeeze()


def total_txs():
    """Total number of unique bitcoin transactions (cumulative).

    URL: https://www.quandl.com/data/BCHAIN/NTRAT"""
    result = quandl.get("BCHAIN/NTRAT")
    result.columns = ['Total txs']
    return result.squeeze()


def estimated_tx_volume_BTC():
    """Similar to the total output volume with the addition of an algorithm which
    attempts to remove change from the total value. This may be a more accurate
    reflection of the true transaction volume.

    URL: https://www.quandl.com/data/BCHAIN/ETRAV"""
    result = quandl.get("BCHAIN/ETRAV")
    result.columns = ['Tx Volume (BTC)']
    return result.squeeze()


def estimated_tx_volume_USD():
    """Similar to the total output volume with the addition of an algorithm which
    attempts to remove change from the total value. This may be a more accurate
    reflection of the true transaction volume.

    URL: https://www.quandl.com/data/BCHAIN/ETRVU"""
    result = quandl.get("BCHAIN/ETRVU")
    result.columns = ['Tx Volume (USD)']
    return result.squeeze()


def output_volume_USD():
    """The total value of all transaction outputs per day. This includes coins
    which were returned to the sender as change

    URL: https://www.quandl.com/data/BCHAIN/TOUTV"""
    result = quandl.get("BCHAIN/TOUTV")
    result.columns = ['Output Volume (USD)']
    return result.squeeze()


# def network_deficit_USD():
#     """Data showing difference between transaction fees and cost of bitcoin mining.
#     It's exactly the same as the negative USD value of the block reward.
#     I won't use it, incomplete data (only until 2016).
#     Block Reward is calculated later, on process_df
#
#     URL: https://www.quandl.com/data/BCHAIN/NETDF"""
#     result = quandl.get("BCHAIN/NETDF")
#     result.columns = ['Network Deficit']
#     return result.squeeze()


def total_BTC_mined():
    """Data showing the historical total number of bitcoins which have been mined.

    URL: https://www.quandl.com/data/BCHAIN/TOTBC"""
    result = quandl.get("BCHAIN/TOTBC")
    result.columns = ['Total BTC mined']
    return result.squeeze()


def mkt_cap_USD():
    """Data showing the total number of bitcoins in circulation the market price in USD.

    URL: https://www.quandl.com/data/BCHAIN/MKTCP"""
    result = quandl.get("BCHAIN/MKTCP")
    result.columns = ['Market Cap (USD)']
    return result.squeeze()


def mempool_tx_count():
    """The number of transactions waiting to be confirmed.

    URL: https://blockchain.info/charts/mempool-count?timespan=all"""
    url = 'https://api.blockchain.info/charts/mempool-count?timespan=all&format=json'
    page = requests.get(url)
    data = page.json()['values']
    df = pd.DataFrame(data)
    df['Date'] = [datetime.datetime.fromtimestamp(d) for d in df.x]
    df.index = pd.to_datetime(df.Date)
    df = df.drop(['Date', 'x'], axis=1)
    df = df.resample('D').sum()
    df.columns = ['Mempool tx count']
    df.fillna(method='pad')
    df.fillna(0)
    return df.squeeze()


def mempool_size_growth_bytes():
    """The rate at which the mempool is growing per second, in bypes per day.

    URL: https://blockchain.info/charts/mempool-growth?timespan=all"""
    url = 'https://api.blockchain.info/charts/mempool-count?timespan=all&format=json'
    page = requests.get(url)
    data = page.json()['values']
    df = pd.DataFrame(data)
    df['Date'] = [datetime.datetime.fromtimestamp(d) for d in df.x]
    df.index = pd.to_datetime(df.Date)
    df = df.drop(['Date', 'x'], axis=1)
    df = df.resample('D').sum()
    df.columns = ['Mempool Growth Rate (bytes/day)']
    df.fillna(method='pad')
    df.fillna(0)
    return df.squeeze()


def mempool_size_bytes():
    """The aggregate size of transactions waiting to be confirmed, in bytes.

    URL: https://blockchain.info/charts/mempool-size?timespan=all"""
    url = 'https://api.blockchain.info/charts/mempool-size?timespan=all&format=json'
    page = requests.get(url)
    data = page.json()['values']
    df = pd.DataFrame(data)
    df['Date'] = [datetime.datetime.fromtimestamp(d) for d in df.x]
    df.index = pd.to_datetime(df.Date)
    df = df.drop(['Date', 'x'], axis=1)
    df = df.resample('D').sum()
    df.columns = ['Mempool Size (bytes)']
    df.fillna(method='pad')
    df.fillna(0)
    return df.squeeze()


def UTXO_count():
    """The number of unspent Bitcoin transactions outputs, also known as the UTXO
    set size.

    URL: https://blockchain.info/charts/utxo-count?timespan=all"""
    url = 'https://api.blockchain.info/charts/utxo-count?timespan=all&format=json'
    page = requests.get(url)
    data = page.json()['values']
    df = pd.DataFrame(data)
    df['Date'] = [datetime.datetime.fromtimestamp(d) for d in df.x]
    df.index = pd.to_datetime(df.Date)
    df = df.drop(['Date', 'x'], axis=1)
    df = df.resample('D').sum()
    df = df.interpolate(method='pchip')
    df.columns = ['UTXO count']
    return df.squeeze()


def difficulty():
    """TA relative measure of how difficult it is to find a new block. The difficulty is
    adjusted periodically as a function of how much hashing power has been deployed by
    the network of miners.

    URL: https://blockchain.info/charts/difficulty?timespan=all"""
    url = 'https://api.blockchain.info/charts/difficulty?timespan=all&format=json'
    page = requests.get(url)
    data = page.json()['values']
    df = pd.DataFrame(data)
    df['Date'] = [datetime.datetime.fromtimestamp(d) for d in df.x]
    df.index = pd.to_datetime(df.Date)
    df = df.drop(['Date', 'x'], axis=1)
    df = df.resample('D').last()
    df = df.fillna(method='pad')
    df.columns = ['Difficulty']
    return df.squeeze()
