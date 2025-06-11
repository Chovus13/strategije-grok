import numpy as np
import pandas as pd
from pandas import DataFrame
import talib
from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from technical.indicators import donchian


class JohnWickSniperStrategy(IStrategy):
    """
    Unaprijeđena verzija JohnWickSniper strategije s ATR stop-loss-om i Volume Spike potvrdom.
    Radi na 15m timeframe-u, koristi 5m/1m za potvrdu, Hammer/Reverse Hammer svijeće,
    Donchian+ADX za LONG, Bollinger+RSI za SHORT.
    """

    # Parametri strategije
    timeframe = "15m"
    minimal_roi = {"0": 0.02}  # 2% ROI
    stoploss = -0.015  # Početni fiksni stop-loss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015

    # Optimizirani parametri
    donchian_period = IntParameter(10, 30, default=20, space="buy")
    adx_period = IntParameter(10, 20, default=14, space="buy")
    adx_threshold = DecimalParameter(20, 40, default=25, space="buy")
    bb_period = IntParameter(10, 30, default=20, space="sell")
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, space="sell")
    rsi_period = IntParameter(10, 20, default=14, space="sell")
    rsi_sell = DecimalParameter(60, 80, default=70, space="sell")
    atr_period = IntParameter(10, 20, default=14, space="buy_sell")
    volume_spike_factor = DecimalParameter(1.5, 3.0, default=2.0, space="buy_sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Dodaje tehničke indikatore u dataframe, uključujući ATR i Volume Spike.
        """
        try:
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

            # ATR za dinamički stop-loss
            dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'],
                                         timeperiod=self.atr_period.value)

            # Volume Spike (volumen veći od proseka za faktor)
            dataframe['vol_avg'] = dataframe['volume'].rolling(window=20).mean()
            dataframe['volume_spike'] = dataframe['volume'] > (dataframe['vol_avg'] * self.volume_spike_factor.value)

            # Hammer i Reverse Hammer svijeće
            dataframe['hammer'] = self.detect_hammer(dataframe)
            dataframe['reverse_hammer'] = self.detect_reverse_hammer(dataframe)

            # Dohvati 5m i 1m podatke
            dataframe_5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe="5m")
            dataframe_1m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe="1m")

            dataframe_5m['hammer_5m'] = self.detect_hammer(dataframe_5m)
            dataframe_5m['reverse_hammer_5m'] = self.detect_reverse_hammer(dataframe_5m)
            dataframe_1m['hammer_1m'] = self.detect_hammer(dataframe_1m)
            dataframe_1m['reverse_hammer_1m'] = self.detect_reverse_hammer(dataframe_1m)

            # Resample na 15m
            dataframe = dataframe.merge(
                dataframe_5m[['date', 'hammer_5m', 'reverse_hammer_5m']].set_index('date'),
                how='left', left_index=True, right_index=True
            )
            dataframe = dataframe.merge(
                dataframe_1m[['date', 'hammer_1m', 'reverse_hammer_1m']].set_index('date'),
                how='left', left_index=True, right_index=True
            )

            dataframe.fillna(method='ffill', inplace=True)
            return dataframe

        except Exception as e:
            print(f"Greška u populate_indicators: {e}")
            return dataframe

    def detect_hammer(self, dataframe: DataFrame) -> pd.Series:
        """Detektuje Hammer svijeću (bullish)."""
        body = abs(dataframe['close'] - dataframe['open'])
        lower_wick = dataframe['open'].where(dataframe['close'] > dataframe['open'], dataframe['close']) - dataframe[
            'low']
        upper_wick = dataframe['high'] - dataframe['close'].where(dataframe['close'] > dataframe['open'],
                                                                  dataframe['open'])
        return (lower_wick > 2 * body) & (upper_wick < 0.5 * body) & (dataframe['close'] > dataframe['open'])

    def detect_reverse_hammer(self, dataframe: DataFrame) -> pd.Series:
        """Detektuje Reverse Hammer svijeću (bearish)."""
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
        # LONG: Breakout + Volume Spike
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['dc_upper'])
                    & (dataframe['adx'] > self.adx_threshold.value)
                    & (dataframe['hammer'])
                    & (dataframe['hammer_5m'] | dataframe['hammer_1m'])
                    & (dataframe['volume_spike'])
            ),
            'enter_long'] = 1

        # SHORT: Mean Reversion + Volume Spike
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['bb_upper'])
                    & (dataframe['rsi'] > self.rsi_sell.value)
                    & (dataframe['reverse_hammer'])
                    & (dataframe['reverse_hammer_5m'] | dataframe['reverse_hammer_1m'])
                    & (dataframe['volume_spike'])
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
                    & (dataframe['reverse_hammer_5m'] | dataframe['reverse_hammer_1m'])
            ),
            'exit_long'] = 1

        # Izlaz iz SHORT-a
        dataframe.loc[
            (
                    (dataframe['hammer'])
                    & (dataframe['hammer_5m'] | dataframe['hammer_1m'])
            ),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dinamički stop-loss baziran na ATR-u.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1]
            atr = last_candle['atr']
            # Stop-loss na 2x ATR ispod cijene ulaza
            stoploss_price = trade.open_rate - (2 * atr)
            stoploss_percentage = (stoploss_price - current_rate) / current_rate
            return stoploss_percentage
        except Exception as e:
            print(f"Greška u custom_stoploss: {e}")
            return self.stoploss

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """Postavlja leverage na 3x."""
        return 3.0

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: 'datetime', **kwargs) -> bool:
        """Provjerava valjanost ulaza u trejd."""
        try:
            return True
        except Exception as e:
            print(f"Greška u confirm_trade_entry: {e}")
            return False