import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from datetime import datetime


class JohnWickSniperStrategy(IStrategy):
    """
    V2 verzija strategije sa ispravljenim indikatorima i signalima
    """

    timeframe = "15m"
    informative_timeframes = ["5m", "1m"]

    minimal_roi = {"0": 0.02}
    stoploss = -0.015
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015

    # Optimizacioni parametri
    donchian_period = IntParameter(10, 30, default=20, space="buy")
    adx_period = IntParameter(10, 20, default=14, space="buy")
    adx_threshold = DecimalParameter(20, 40, default=25, space="buy")
    bb_period = IntParameter(10, 30, default=20, space="sell")
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, space="sell")
    rsi_period = IntParameter(10, 20, default=14, space="sell")
    rsi_sell = DecimalParameter(60, 80, default=70, space="sell")

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, tf) for pair in pairs for tf in self.informative_timeframes]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            # Donchian Channel
            period = int(self.donchian_period.value)
            dataframe['dc_upper'] = dataframe['high'].rolling(window=period).max()
            dataframe['dc_lower'] = dataframe['low'].rolling(window=period).min()

            # ADX
            dataframe['adx'] = ta.ADX(dataframe, timeperiod=int(self.adx_period.value))

            # Bollinger Bands
            bb = ta.BBANDS(dataframe, timeperiod=int(self.bb_period.value),
                           nbdevup=float(self.bb_std.value), nbdevdn=float(self.bb_std.value))
            dataframe['bb_upper'] = bb['upperband']
            dataframe['bb_middle'] = bb['middleband']
            dataframe['bb_lower'] = bb['lowerband']

            # RSI
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=int(self.rsi_period.value))

            # Hammer i Reverse Hammer detekcija
            dataframe['hammer'] = self.detect_hammer(dataframe)
            dataframe['reverse_hammer'] = self.detect_reverse_hammer(dataframe)

            # Informativni timeframe-ovi - ISPRAVLJENA VERZIJA
            for tf in self.informative_timeframes:
                informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf)

                # Resetuj indeks da bismo imali 'date' kolonu
                informative = informative.reset_index()

                # Detekcija formacija
                informative['hammer'] = self.detect_hammer(informative)
                informative['reverse_hammer'] = self.detect_reverse_hammer(informative)

                # Selektuj samo potrebne kolone
                informative = informative[['date', 'hammer', 'reverse_hammer']].copy()

                # Preimenuj kolone sa timeframe sufiksom
                informative.columns = ['date', f'hammer_{tf}', f'reverse_hammer_{tf}']

                # Spoji sa glavnim dataframe-om
                dataframe = dataframe.merge(informative, on='date', how='left')

            print(f"BTC/USDT Poslednji candle:")
            print(f"Close: {dataframe['close'].iloc[-1]}")
            print(f"DC Upper: {dataframe['dc_upper'].iloc[-1]}")
            print(f"ADX: {dataframe['adx'].iloc[-1]}")
            print(f"RSI: {dataframe['rsi'].iloc[-1]}")
            print(f"BB Upper: {dataframe['bb_upper'].iloc[-1]}")
            print(f"Volume: {dataframe['volume'].iloc[-1]}")
            print(f"Avg Volume: {dataframe['volume'].rolling(20).mean().iloc[-1]}")

            lev = self.leverage(metadata['pair'], datetime.now(), dataframe['close'].iloc[-1],
                                3, 5, 'long' if dataframe['hammer'].iloc[-1] else 'short')

            print(f"Calculated leverage for {metadata['pair']}: {lev}x")

            if self.dp.runmode.value in ('backtest', 'hyperopt'):
                dataframe['funding_rate'] = 0  # Mock za backtest
            else:
                # Uzmi stvarne funding rate-ove za futures
                fr = self.dp.exchange.get_funding_rate(metadata['pair'])
                dataframe['funding_rate'] = fr['rate'] if fr else 0

            return dataframe

        except Exception as e:
            print(f"Greška u populate_indicators: {str(e)}")
            return dataframe

    @staticmethod
    def detect_hammer(df: DataFrame) -> pd.Series:
        """Detekcija Hammer formacije"""
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        return (lower_wick > 2 * body) & (upper_wick < 0.5 * body) & (df['close'] > df['open'])

    @staticmethod
    def detect_reverse_hammer(df: DataFrame) -> pd.Series:
        """Detekcija Reverse Hammer formacije"""
        body = abs(df['close'] - df['open'])
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        return (upper_wick > 2 * body) & (lower_wick < 0.5 * body) & (df['close'] < df['open'])

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            # Olabavljeni LONG uslovi
            dataframe.loc[
                (dataframe['close'] > dataframe['dc_upper']) &
                (dataframe['adx'] > 15) &  # Smanjen threshold
                (
                        (dataframe['hammer'] == True) |
                        (dataframe['hammer_5m'] == True) |
                        (dataframe['hammer_1m'] == True)
                ) &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 0.8),  # Volume filter
                'enter_long'] = 1

            # Olabavljeni SHORT uslovi
            dataframe.loc[
                (dataframe['close'] > dataframe['bb_upper']) &
                (dataframe['rsi'] > 65) &  # Smanjen threshold
                (
                        (dataframe['reverse_hammer'] == True) |
                        (dataframe['reverse_hammer_5m'] == True) |
                        (dataframe['reverse_hammer_1m'] == True)
                ) &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 0.8),
                'enter_short'] = 1

            return dataframe
        except Exception as e:
            print(f"Greška u populate_entry_trend: {str(e)}")
            return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        # Provera futures uslova
        if not pair.endswith(":USDT"):
            return False

        # Provera likvidnosti
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if abs(dataframe['funding_rate'].iloc[-1]) > 0.0005:  # > 0.05%
            return False

        return True

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Izlaz iz LONG-a
        dataframe.loc[
            (dataframe['reverse_hammer'] == True) &
            ((dataframe['reverse_hammer_5m'] == True) | (dataframe['reverse_hammer_1m'] == True)),
            'exit_long'] = 1

        # Izlaz iz SHORT-a
        dataframe.loc[
            (dataframe['hammer'] == True) &
            ((dataframe['hammer_5m'] == True) | (dataframe['hammer_1m'] == True)),
            'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Dinamičko podešavanje leverage-a za futures trading
        """
        # Proveri da li pair podržava leverage
        if "USDT" not in pair:
            return 1.0  # Za spot parove

        # Dinamički leverage based na volatilnosti
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 20:
            return min(3.0, max_leverage)

        # Izračunaj ATR za volatilnost
        atr = ta.ATR(dataframe, timeperiod=14).iloc[-1]
        current_price = dataframe['close'].iloc[-1]
        atr_pct = (atr / current_price) * 100

        # Podesi leverage based na volatilnosti
        if atr_pct < 1.5:
            leverage = min(5.0, max_leverage)
        elif atr_pct < 3.0:
            leverage = min(3.0, max_leverage)
        else:
            leverage = min(1.5, max_leverage)

        return leverage