import numpy as np
import pandas as pd
import pandas_ta as pta
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class JohnWickSniperStrategy(IStrategy):
    """
    V1 verzija scalping strategije za Freqtrade na 15m timeframe-u.
    Koristi 5m i 1m kandele za potvrdu trenda, Hammer/Reverse Hammer svijeće,
    Donchian+ADX za LONG, Bollinger+RSI za SHORT.
    """

    # Parametri strategije
    timeframe = "15m"
    informative_timeframes = ["5m", "1m"]

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

    # Leverage parametri
    leverage_num = IntParameter(1, 10, default=3, space="protection")
    margin_mode = CategoricalParameter(['isolated', 'cross'], default='isolated', space="protection")

    # def informative_pairs(self):
    # """
    # Definiše informativne timeframe-ove za 5m i 1m.
    # """
    # pairs = self.dp.current_whitelist()
    # informative_pairs = [(pair, "5m") for pair in pairs] + [(pair, "1m") for pair in pairs]
    # return informative_pairs

    # def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    #     # Informative dataframe-ovi za 5m i 1m
    #     # Ova logika ostaje ista
    #     for timeframe in self.informative_timeframes:
    #         informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=timeframe)
    #         # Ovde dodajemo sufiks da bismo razlikovali kolone (npr. close_5m)
    #         # Ovo je dobra praksa, ali merge_informative_pair to radi automatski
    #         dataframe = merge_informative_pair(dataframe, informative, self.timeframe, timeframe, ffill=True)
    #
    #         # Donchian Channel - ISPRAVLJENO KORIŠĆENJEM PANDAS-TA
    #         # pta.donchian() vraća DataFrame sa kolonama npr. 'DCU_20', 'DCL_20'
    #         # Moramo ih preimenovati ili direktno dodeliti.
    #         donchian_df = pta.donchian(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'],
    #                                    length=self.donchian_period.value)
    #
    #         # Nazivi kolona koje generiše pandas-ta zavise od perioda (npr. DCU_20)
    #         # Zato koristimo f-string da dinamički kreiramo naziv.
    #         upper_channel_name = f'DCU_{self.donchian_period.value}'
    #         lower_channel_name = f'DCL_{self.donchian_period.value}'
    #
    #         # Dodajemo kolone u glavni dataframe sa nazivima koje strategija očekuje
    #         dataframe['dc_upper'] = donchian_df[upper_channel_name]
    #         dataframe['dc_lower'] = donchian_df[lower_channel_name]
    #
    #         # return dataframe
    #         # if dataframe['dc_upper'].isna().all() or dataframe['dc_lower'].isna().all():
    #         # print(f"WARNING: Donchian Channels nisu ispravno izračunati za {metadata['pair']}")
    #
    #         # ADX za snagu trenda
    #         dataframe['adx'] = talib.ADX(dataframe['high'], dataframe['low'], dataframe['close'],
    #                                      timeperiod=self.adx_period.value)
    #
    #         # Bollinger Bands za SHORT
    #         dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = talib.BBANDS(
    #             dataframe['close'], timeperiod=self.bb_period.value, nbdevup=self.bb_std.value,
    #             nbdevdn=self.bb_std.value
    #         )
    #
    #         # RSI za prekupljenost
    #         dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=self.rsi_period.value)
    #
    #         # ATR za dinamički stop-loss
    #         dataframe['atr'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'],
    #                                      timeperiod=self.atr_period.value)
    #
    #         # Volume Spike (volumen veći od proseka za faktor)
    #         dataframe['vol_avg'] = dataframe['volume'].rolling(window=20).mean()
    #         dataframe['volume_spike'] = dataframe['volume'] > (dataframe['vol_avg'] * self.volume_spike_factor.value)
    #
    #         # Hammer i Reverse Hammer svijeće
    #         dataframe['hammer'] = self.detect_hammer(dataframe)
    #         dataframe['reverse_hammer'] = self.detect_reverse_hammer(dataframe)
    #
    #         # Dohvati 5m i 1m podatke
    #         dataframe_5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe="5m")
    #         dataframe_1m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe="1m")
    #
    #         dataframe_5m['hammer_5m'] = self.detect_hammer(dataframe_5m)
    #         dataframe_5m['reverse_hammer_5m'] = self.detect_reverse_hammer(dataframe_5m)
    #         dataframe_1m['hammer_1m'] = self.detect_hammer(dataframe_1m)
    #         dataframe_1m['reverse_hammer_1m'] = self.detect_reverse_hammer(dataframe_1m)
    #
    #         # Resample na 15m
    #         dataframe = dataframe.merge(
    #             dataframe_5m[['date', 'hammer_5m', 'reverse_hammer_5m']].set_index('date'),
    #             how='left', left_index=True, right_index=True
    #         )
    #         dataframe = dataframe.merge(
    #             dataframe_1m[['date', 'hammer_1m', 'reverse_hammer_1m']].set_index('date'),
    #             how='left', left_index=True, right_index=True
    #         )
    #
    #         dataframe.fillna(method='ffill', inplace=True)
    #
    #         return dataframe
    #         # if dataframe['dc_upper'].isna().all() or dataframe['dc_lower'].isna().all():
    #         print(f"WARNING: Donchian Channels nisu ispravno izračunati za {metadata['pair']}")
    #
    #         return dataframe
    # Ne zaboravite da importujete pandas na vrhu fajla ako već niste
    # import pandas as pd

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- KORAK 1: Izračunavanje indikatora na informativnim timeframe-ovima ---

        for timeframe_inf in self.informative_timeframes:
            # Preuzimanje informativnog dataframe-a (npr. za '5m' ili '1m')
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=timeframe_inf)

            # Na ovom informativnom dataframe-u izračunavamo indikatore koji su nam potrebni
            # Primer za hammer sveću - prilagodite ovo vašim indikatorima
            informative['hammer'] = self.heikin_ashi(informative).apply(
                lambda x: 1 if (x['ha_high'] - x['ha_low']) > 3 * abs(x['ha_open'] - x['ha_close']) and
                               (x['ha_close'] - x['ha_low']) / (0.001 + x['ha_high'] - x['ha_low']) > 0.6 and
                               (x['ha_open'] - x['ha_low']) / (0.001 + x['ha_high'] - x['ha_low']) > 0.6
                else 0, axis=1
            )
            # Primer za volume spike - prilagodite ovo vašim indikatorima
            informative['volume_spike'] = (informative['volume'] > informative['volume'].rolling(20).mean() * 2).astype(
                int)

            # Preimenovanje kolona da bismo znali sa kog su timeframe-a
            # Npr. 'hammer' postaje 'hammer_5m'
            informative.rename(columns={
                'hammer': f'hammer_{timeframe_inf}',
                'volume_spike': f'volume_spike_{timeframe_inf}'
            }, inplace=True)

            # --- KORAK 2: Ispravno spajanje samo potrebnih kolona ---
            # Koristimo merge_asof da bezbedno dodamo signale iz bržeg u sporiji timeframe
            # On pronalazi poslednju dostupnu vrednost iz 'informative' za svaku sveću u 'dataframe'
            dataframe = pd.merge_asof(
                dataframe,
                informative[[
                    'date',
                    f'hammer_{timeframe_inf}',
                    f'volume_spike_{timeframe_inf}'
                ]],
                on='date',
                direction='backward'
            )

        # --- KORAK 3: Izračunavanje indikatora na glavnom (15m) timeframe-u ---
        # Sada kada su informativni podaci spojeni, računamo indikatore za 15m

        # Donchian Channel za LONG
        donchian_df = pta.donchian(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'],
                                   length=self.donchian_period.value)
        upper_channel_name = f'DCU_{self.donchian_period.value}'
        lower_channel_name = f'DCL_{self.donchian_period.value}'
        dataframe['dc_upper'] = donchian_df[upper_channel_name]
        dataframe['dc_lower'] = donchian_df[lower_channel_name]

        # ADX za snagu trenda
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)

        # Hammer i Volume spike na glavnom timeframe-u
        dataframe['hammer'] = self.heikin_ashi(dataframe).apply(
            lambda x: 1 if (x['ha_high'] - x['ha_low']) > 3 * abs(x['ha_open'] - x['ha_close']) and
                           (x['ha_close'] - x['ha_low']) / (0.001 + x['ha_high'] - x['ha_low']) > 0.6 and
                           (x['ha_open'] - x['ha_low']) / (0.001 + x['ha_high'] - x['ha_low']) > 0.6
            else 0, axis=1
        )
        dataframe['volume_spike'] = (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2).astype(int)

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

