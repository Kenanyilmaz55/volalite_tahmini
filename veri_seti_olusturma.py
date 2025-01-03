import pandas as pd
import numpy as np
import requests
import csv
from datetime import datetime
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

def fetch_binance_data():
    """
    Binance API'sinden veri çeker ve bir DataFrame döndürür.

    Returns:
        pd.DataFrame: Çekilen Binance verileri
    """
    # Binance API endpoint'leri
    KLINE_ENDPOINT = "https://api.binance.com/api/v3/klines"
    TICKER_ENDPOINT = "https://api.binance.com/api/v3/ticker/24hr"

    # Parametreler
    SYMBOL = "BTCUSDT"
    INTERVAL = "1h"
    LIMIT = 1000

    # Kline verilerini çek
    print("Kline verileri çekiliyor...")
    kline_params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT
    }
    kline_response = requests.get(KLINE_ENDPOINT, params=kline_params)
    kline_data = kline_response.json()

    # 24h Ticker verilerini çek
    print("24h Ticker verileri çekiliyor...")
    ticker_response = requests.get(TICKER_ENDPOINT, params={"symbol": SYMBOL})
    ticker_data = ticker_response.json()

    # Verileri DataFrame'e dönüştür
    data = []
    for kline in kline_data:
        open_time = datetime.utcfromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        close_time = datetime.utcfromtimestamp(kline[6] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        data.append([
            open_time, kline[1], kline[2], kline[3], kline[4], kline[5], close_time,
            kline[7], kline[8], ticker_data.get("priceChange", "N/A"),
            ticker_data.get("priceChangePercent", "N/A"),
            ticker_data.get("highPrice", "N/A"),
            ticker_data.get("lowPrice", "N/A"),
            ticker_data.get("weightedAvgPrice", "N/A")
        ])

    columns = [
        "Open Time", "Open Price", "High Price", "Low Price", "Close Price", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades", "24h Price Change",
        "24h Price Change Percent", "24h High Price", "24h Low Price", "Weighted Average Price"
    ]
    df = pd.DataFrame(data, columns=columns)

    # Tip dönüşümleri
    numeric_columns = ["Open Price", "High Price", "Low Price", "Close Price", "Volume"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    return df

def add_hourly_volatility_features(df):
    """
    Saatlik periyotlar için volatilite analizi özellikleri ekler.

    Parameters:
    df (pd.DataFrame): İşlenecek DataFrame

    Returns:
    pd.DataFrame: Yeni özellikler eklenmiş DataFrame
    """
    df = df.copy()

    # Temel volatilite göstergeleri
    df['Hourly_Return'] = df['Close Price'].pct_change()
    df['Hourly_Range'] = df['High Price'] - df['Low Price']
    df['Range_Percentage'] = (df['High Price'] - df['Low Price']) / df['Open Price'] * 100

    # Kısa vadeli volatilite hesaplamaları
    hours = [4, 8, 24]
    for hour in hours:
        df[f'Volatility_{hour}h'] = df['Hourly_Return'].rolling(window=hour).std()
        df[f'TR'] = np.maximum(
            df['High Price'] - df['Low Price'],
            np.maximum(
                abs(df['High Price'] - df['Close Price'].shift(1)),
                abs(df['Low Price'] - df['Close Price'].shift(1))
            )
        )
        df[f'ATR_{hour}h'] = df['TR'].rolling(window=hour).mean()
        bb = BollingerBands(close=df['Close Price'], window=hour)
        df[f'BB_Upper_{hour}h'] = bb.bollinger_hband()
        df[f'BB_Lower_{hour}h'] = bb.bollinger_lband()
        df[f'BB_Width_{hour}h'] = (df[f'BB_Upper_{hour}h'] - df[f'BB_Lower_{hour}h']) / df['Close Price']
        df[f'EMA_{hour}h'] = EMAIndicator(close=df['Close Price'], window=hour).ema_indicator()

    # RSI - farklı periyotlar için
    for period in [6, 14, 24]:
        rsi = RSIIndicator(close=df['Close Price'], window=period)
        df[f'RSI_{period}h'] = rsi.rsi()

    # MACD
    macd = MACD(close=df['Close Price'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # Stochastic Oscillator
    stoch = StochasticOscillator(
        high=df['High Price'], low=df['Low Price'], close=df['Close Price'], window=14, smooth_window=3
    )
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # Hacim bazlı göstergeler
    df['Volume_MA4'] = df['Volume'].rolling(window=4).mean()
    df['Volume_MA8'] = df['Volume'].rolling(window=8).mean()
    df['Volume_MA24'] = df['Volume'].rolling(window=24).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA24']

    # VWAP
    vwap = VolumeWeightedAveragePrice(
        high=df['High Price'], low=df['Low Price'], close=df['Close Price'], volume=df['Volume'], window=24
    )
    df['VWAP'] = vwap.volume_weighted_average_price()
    df['VWAP_Distance'] = ((df['Close Price'] - df['VWAP']) / df['VWAP']) * 100
    return df

if __name__ == "__main__":
    print("Binance API'den veri çekiliyor ve işleniyor...")
    market_data = fetch_binance_data()
    market_data_with_features = add_hourly_volatility_features(market_data)

    output_filename = f"btc_volatility_features_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    market_data_with_features.to_csv(output_filename, index=False)
    print(f"Veriler başarıyla {output_filename} dosyasına kaydedildi.")
