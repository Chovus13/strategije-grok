import numpy as np
import pandas as pd
from pandas import DataFrame
import talib
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy import StoplossGuard
from technical.indicators import donchian
from datetime import datetime


class JohnWickSniperStrategy(IStrategy):
    #

    # Osnovni parametri

    timeframe = "15m"
    informative_timeframes = ["5m", "1m"]

    minimal_roi = {"0": 0.02}  # 2% ROI
    stoploss = -0.015  # Fiksni stop-loss od 1.5%

    trailing_stop = True
    trailing_stop_positive = 0.01  # 1% pozitivni offset
    trailing_stop_positive_offset = 0.015  # 1.5% za pokretanje trailing stop-a

    # Optimizirani parametri
    donchian_period = IntParameter(10, 30, default=20, space="buy")
    adx_period = IntParameter(10, 20, default=14, space="buy")
    adx_threshold = DecimalParameter(20, 40, default=25, space="buy")
    bb_period = IntParameter(10, 30, default=20, space="sell")
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, space="sell")
    rsi_period = IntParameter(10, 20, default=14, space="sell")
    rsi_sell = DecimalParameter(60, 80, default=70, space="sell")

    def informative_pairs(self):
        """
        Definiše informativne timeframe-ove za 5m i 1m.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "5m") for pair in pairs] + [(pair, "1m") for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Dodaje tehničke indikatore u dataframe.
        """
        try:
            # Informative dataframe-ovi za 5m i 1m
            for timeframe in self.informative_timeframes:
                informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=timeframe)
                dataframe = merge_informative_pair(dataframe, informative, timeframe, 1, ffill=True)

            # Donchian Channel za LONG
            donchian_channel = donchian(dataframe, period=self.donchian_period.value)
            dataframe['dc_upper'] = donchian_channel['upper']
            dataframe['dc_lower'] = donchian_channel['lower']

            # ADX za snagu trenda
            dataframe['adx'] = talib.ADX(dataframe['high'], dataframe['low'], dataframe['close'],
                                         timeperiod=self.adx_period.value)

            # Bollinger Bands za SHORT
            dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = talib.BBANDS(
                dataframe['close'], timeperiod=self.bb_period.value, nbdevup=self.bb_std.value,
                nbdevdn=self.bb_std.value
            )

            # RSI za prekupljenost
            dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=self.rsi_period.value)

            # Hammer i Reverse Hammer svijeće na 15m
            dataframe['hammer'] = self.detect_hammer(dataframe)
            dataframe['reverse_hammer'] = self.detect_reverse_hammer(dataframe)

            # Hammer i Reverse Hammer na 5m i 1m
            dataframe['hammer_5m'] = self.detect_hammer(dataframe.select_plottable().copy())
            dataframe['reverse_hammer_5m'] = self.detect_reverse_hammer(dataframe.select_plottable().copy())
            dataframe['hammer_1m'] = self.detect_hammer(dataframe.select_plottable().copy())
            dataframe['reverse_hammer_1m'] = self.detect_reverse_hammer(dataframe.select_plottable().copy())

            return dataframe

        except Exception as e:
            print(f"Greška u populate_indicators: {e}")
            return dataframe

    # @staticmethod
    def detect_hammer(dataframe: DataFrame) -> pd.Series:
        """Detektuje Hammer svijeću (bullish) kao statičku metodu."""
        body = abs(dataframe['close'] - dataframe['open'])
        lower_wick = dataframe['open'].where(dataframe['close'] > dataframe['open'], dataframe['close']) - dataframe[
            'low']
        upper_wick = dataframe['high'] - dataframe['close'].where(dataframe['close'] > dataframe['open'],
                                                                  dataframe['open'])
        return (lower_wick > 2 * body) & (upper_wick < 0.5 * body) & (dataframe['close'] > dataframe['open'])

    # @staticmethod
    def detect_reverse_hammer(dataframe: DataFrame) -> pd.Series:
        """Detektuje Reverse Hammer svijeću (bearish) kao statičku metodu."""
        body = abs(dataframe['close'] - dataframe['open'])
        upper_wick = dataframe['high'] - dataframe['close'].where(dataframe['close'] > dataframe['open'],
                                                                  dataframe['open'])
        lower_wick = dataframe['open'].where(dataframe['close'] > dataframe['open'], dataframe['close']) - dataframe[
            'low']
        return (upper_wick > 2 * body) & (lower_wick < 0.5 * body) & (dataframe['close'] < dataframe['open'])

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Uslovi za ulaz u LONG i SHORT pozicije.
        """
        # LONG: Breakout + Hammer potvrda
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['dc_upper'])
                    & (dataframe['adx'] > self.adx_threshold.value)
                    & (dataframe['hammer'])
                    & ((dataframe['hammer_5m']) | (dataframe['hammer_1m']))
            ),
            'enter_long'] = 1

        # SHORT: Mean Reversion + Reverse Hammer potvrda
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['bb_upper'])
                    & (dataframe['rsi'] > self.rsi_sell.value)
                    & (dataframe['reverse_hammer'])
                    & ((dataframe['reverse_hammer_5m']) | (dataframe['reverse_hammer_1m']))
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Uslovi za izlaz iz LONG i SHORT pozicija.
        """
        # Izlaz iz LONG-a
        dataframe.loc[
            (
                    (dataframe['reverse_hammer'])
                    & ((dataframe['reverse_hammer_5m']) | (dataframe['reverse_hammer_1m']))
            ),
            'exit_long'] = 1

        # Izlaz iz SHORT-a
        dataframe.loc[
            (
                    (dataframe['hammer'])
                    & ((dataframe['hammer_5m']) | (dataframe['hammer_1m']))
            ),
            'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """Postavlja leverage na 3x za scalping."""
        return 3.0

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        """Provjerava valjanost ulaza u trejd."""
        try:
            return True
        except Exception as e:
            print(f"Greška u confirm_trade_entry: {e}")
            return False

# Dodaj logger ako želiš (opciono za V1)
# import logging

# logger = logging.getLogger(__name__)