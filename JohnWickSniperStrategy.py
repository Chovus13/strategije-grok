import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from freqtrade.strategy import (CategoricalParameter, DecimalParameter,
                                IntParameter)
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class JohnWickSniperStrategy(IStrategy):
    """
    V4 verzija sa ispravljenim USDT futures podrškom i leverage sistemom
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

    # Leverage parametri
    leverage_num = IntParameter(1, 10, default=3, space="protection")
    margin_mode = CategoricalParameter(['isolated', 'cross'],
                                       default='isolated', space="protection")

    def informative_pairs(self):
        pairs = [p for p in self.dp.current_whitelist() if ":USDT" in p]
        return [(pair, tf) for pair in pairs for tf in self.informative_timeframes]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if ":USDT" not in metadata['pair']:
            return dataframe

        try:
            # Donchian Channel
            period = int(self.donchian_period.value)
            dataframe['dc_upper'] = dataframe['high'].rolling(window=period).max()
            dataframe['dc_lower'] = dataframe['low'].rolling(window=period).min()

            # ADX
            dataframe['adx'] = ta.ADX(dataframe, timeperiod=int(self.adx_period.value))

            # Bollinger Bands
            bb = ta.BBANDS(dataframe, timeperiod=int(self.bb_period.value),
                           nbdevup=float(self.bb_std.value),
                           nbdevdn=float(self.bb_std.value))
            dataframe['bb_upper'] = bb['upperband']
            dataframe['bb_middle'] = bb['middleband']
            dataframe['bb_lower'] = bb['lowerband']

            # RSI
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=int(self.rsi_period.value))

            # Hammer i Reverse Hammer
            dataframe['hammer'] = self.detect_hammer(dataframe)
            dataframe['reverse_hammer'] = self.detect_reverse_hammer(dataframe)

            # Informativni timeframe-ovi
            for tf in self.informative_timeframes:
                informative = self.dp.get_pair_dataframe(
                    pair=metadata['pair'], timeframe=tf)

                informative['hammer'] = self.detect_hammer(informative)
                informative['reverse_hammer'] = self.detect_reverse_hammer(informative)

                informative = informative[['date', 'hammer', 'reverse_hammer']].copy()
                informative.columns = ['date', f'hammer_{tf}', f'reverse_hammer_{tf}']
                dataframe = dataframe.merge(informative, on='date', how='left')

            return dataframe

        except Exception as e:
            logger.error(f"Greška u indikatorima za {metadata['pair']}: {e}")
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

        except Exception as e:
            print(f"Greška u populate_entry_trend: {str(e)}")
            return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return min(int(self.leverage_num.value), max_leverage)

    def adjust_leverage(self, pair: str, side: str):
        try:
            exchange = self.dp._exchange
            leverage = int(self.leverage_num.value)
            margin_mode = str(self.margin_mode.value).lower()

            exchange.set_margin_mode(margin_mode, pair)
            exchange.set_leverage(leverage, pair)

            logger.info(f"Leverage postavljen: {margin_mode} {leverage}x za {pair}")

        except Exception as e:
            logger.error(f"Greška u leverage za {pair}: {e}")

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        self.adjust_leverage(pair, side)
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
