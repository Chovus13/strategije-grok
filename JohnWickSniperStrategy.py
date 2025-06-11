# --- Imports ---
import logging
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# --- Freqtrade strategija ---
from freqtrade.strategy import IStrategy, merge_informative_pair

# --- Indikatori ---
# Pretpostavka je da 'hammer' indikator dolazi iz fajla u 'user_data/technical/indicators'
# Ukoliko nije tu, potrebno ga je dodati.
# from technical.indicators import hammer # <-- Uverite se da je ovaj fajl dostupan Freqtrade-u

# Inicijalizacija loggera
logger = logging.getLogger(__name__)


# Definicija Hammer indikatora (ako 'technical.indicators' nije dostupan)
# Ukoliko imate fajl 'technical/indicators/hammer.py', obrišite ili komentarišite ovu funkciju.
def hammer(dataframe: DataFrame, MAs=False) -> DataFrame:
    """
    Prepoznaje Hammer i Inverted Hammer (Shooting Star) sveće
    Vraća:
       1 za Hammer
      -1 za Inverted Hammer (Shooting Star)
       0 za ostalo
    """
    df = dataframe.copy()

    df["body"] = abs(df["open"] - df["close"])
    df["range"] = df["high"] - df["low"]

    # Uslovi za Hammer
    hammer_candle = (
            (df["body"] > 0.01 * df["range"]) &  # Telo mora postojati
            ((df["high"] - df["close"]) > 2 * df["body"]) &  # Gornji fitilj
            ((df["open"] - df["low"]) < 0.8 * df["body"])  # Donji fitilj mali
    )

    # Uslovi za Inverted Hammer (Shooting Star)
    inverted_hammer_candle = (
            (df["body"] > 0.01 * df["range"]) &  # Telo mora postojati
            ((df["high"] - df["open"]) < 0.8 * df["body"]) &  # Gornji fitilj mali
            ((df["close"] - df["low"]) > 2 * df["body"])  # Donji fitilj
    )

    df.loc[hammer_candle, "hammer"] = 1
    df.loc[inverted_hammer_candle, "hammer"] = -1
    df["hammer"].fillna(0, inplace=True)

    return df["hammer"]


class JohnWickSniperStrategy(IStrategy):
    """
    John Wick Sniper Strategy
    Verzija: 1.1
    Autor: Chovus13 (Modifikovao: Gemini)
    Opis:
    Strategija kombinuje breakout (Donchian) i mean-reversion (Bollinger Bands)
    sa potvrdama na osnovu Hammer/Inverted Hammer sveća na više timeframe-ova.
    """
    # ROI tabela:
    minimal_roi = {
        "0": 0.02  # 2% ROI
    }

    # Stoploss:
    stoploss = -0.015  # 1.5% stoploss

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Glavni Timeframe
    timeframe = '15m'

    def informative_pairs(self):
        """Definiše dodatne (informativne) timeframe-ove koji se koriste za potvrdu."""
        # Uklonjen 1h jer se ne koristi u logici
        return [(self.config['stake_currency'], self.config['base_currency'], "5m"),
                (self.config['stake_currency'], self.config['base_currency'], "1m")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Popunjava dataframe sa svim potrebnim indikatorima."""
        logger.info(f"--- Obračunavam indikatore za {metadata['pair']} ---")

        try:
            # --- Informativni Timeframe-ovi (5m i 1m) ---
            # 5m timeframe za Hammer potvrdu
            informative_5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='5m')
            informative_5m['hammer'] = hammer(informative_5m)
            dataframe = merge_informative_pair(dataframe, informative_5m, self.timeframe, '5m', ffill=True)
            logger.info("✓ Indikator 'hammer_5m' uspešno obračunat i spojen.")

            # 1m timeframe za Hammer potvrdu
            informative_1m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1m')
            informative_1m['hammer'] = hammer(informative_1m)
            dataframe = merge_informative_pair(dataframe, informative_1m, self.timeframe, '1m', ffill=True)
            logger.info("✓ Indikator 'hammer_1m' uspešno obračunat i spojen.")

            # --- Indikatori na glavnom (15m) timeframe-u ---
            # Donchian Channels za LONG breakout
            dataframe['donchian_upper'] = ta.MAX(dataframe['high'], timeperiod=20)
            dataframe['donchian_lower'] = ta.MIN(dataframe['low'], timeperiod=20)
            logger.info("✓ Donchian kanali (20) uspešno obračunati.")

            # ADX za snagu trenda
            dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
            logger.info("✓ ADX (14) uspešno obračunat.")

            # Bollinger Bands za SHORT mean reversion
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.0)
            dataframe['bb_upperband'] = bollinger['upper']
            logger.info("✓ Bollinger Bands (20, 2.0) uspešno obračunati.")

            # RSI za prekupljenost
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            logger.info("✓ RSI (14) uspešno obračunat.")

            # Hammer za 15m timeframe (glavni)
            dataframe['hammer'] = hammer(dataframe)
            logger.info("✓ Hammer (15m) uspešno obračunat.")

        except Exception as e:
            logger.error(f"X Greška prilikom obračuna indikatora za {metadata['pair']}: {e}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Definiše uslove za LONG (buy) i SHORT (sell) ulaze."""

        # --- LONG (BUY) Uslovi ---
        long_conditions = []
        logger.info(f"--- Proveravam LONG uslove za {metadata['pair']} ---")

        # Uslov 1: Cena probija gornju liniju Donchian kanala od prethodne sveće
        c1 = dataframe['close'] > dataframe['donchian_upper'].shift(1)
        long_conditions.append(c1)
        logger.info(f"Proboj Donchian kanala: {'✓' if c1.any() else 'X'}")

        # Uslov 2: Trend je jak (ADX > 25)
        c2 = dataframe['adx'] > 25
        long_conditions.append(c2)
        logger.info(f"ADX > 25: {'✓' if c2.any() else 'X'}")

        # Uslov 3: Hammer sveća na 15m timeframe-u
        # ISPRAVKA: Korišćen je 'hammer_15m' umesto 'hammer'
        c3 = dataframe['hammer'] > 0
        long_conditions.append(c3)
        logger.info(f"Hammer na 15m: {'✓' if c3.any() else 'X'}")

        # Uslov 4: Potvrda sa Hammer svećom na 5m ili 1m
        c4 = (dataframe['hammer_5m'] > 0) | (dataframe['hammer_1m'] > 0)
        long_conditions.append(c4)
        logger.info(f"Hammer potvrda (5m ili 1m): {'✓' if c4.any() else 'X'}")

        if all(cond.any() for cond in long_conditions):
            dataframe.loc[
                (c1 & c2 & c3 & c4),
                'enter_long'] = 1
            logger.info(f"✓ SVI LONG USLOVI ISPUNJENI za {metadata['pair']}")

        # --- SHORT (SELL) Uslovi ---
        short_conditions = []
        logger.info(f"--- Proveravam SHORT uslove za {metadata['pair']} ---")

        # Uslov 1: Cena je iznad gornje Bollinger trake
        c1_short = dataframe['close'] > dataframe['bb_upperband']
        short_conditions.append(c1_short)
        logger.info(f"Cena > BB gornja traka: {'✓' if c1_short.any() else 'X'}")

        # Uslov 2: RSI ukazuje na prekupljenost (>70)
        c2_short = dataframe['rsi'] > 70
        short_conditions.append(c2_short)
        logger.info(f"RSI > 70: {'✓' if c2_short.any() else 'X'}")

        # Uslov 3: Reverse Hammer (Shooting Star) na 15m timeframe-u
        # ISPRAVKA: Korišćen je 'hammer_15m' umesto 'hammer'
        c3_short = dataframe['hammer'] < 0
        short_conditions.append(c3_short)
        logger.info(f"Reverse Hammer na 15m: {'✓' if c3_short.any() else 'X'}")

        # Uslov 4: Potvrda sa Reverse Hammer svećom na 5m ili 1m
        c4_short = (dataframe['hammer_5m'] < 0) | (dataframe['hammer_1m'] < 0)
        short_conditions.append(c4_short)
        logger.info(f"Reverse Hammer potvrda (5m ili 1m): {'✓' if c4_short.any() else 'X'}")

        if all(cond.any() for cond in short_conditions):
            dataframe.loc[
                (c1_short & c2_short & c3_short & c4_short),
                'enter_short'] = 1
            logger.info(f"✓ SVI SHORT USLOVI ISPUNJENI za {metadata['pair']}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Definiše uslove za izlaz iz LONG i SHORT pozicija."""

        # --- Izlaz iz LONG pozicije ---
        # Uslov: Pojavljuje se Reverse Hammer na 15m sa potvrdom na 5m ili 1m
        # ISPRAVKA: Preimenovano u 'populate_exit_buy' i ispravljen naziv kolone
        exit_long_c1 = dataframe['hammer'] < 0
        exit_long_c2 = (dataframe['hammer_5m'] < 0) | (dataframe['hammer_1m'] < 0)

        dataframe.loc[
            (exit_long_c1 & exit_long_c2),
            'exit_long'] = 1

        if (exit_long_c1 & exit_long_c2).any():
            logger.info(f"✓ Uslov za IZLAZ IZ LONG pozicije ispunjen za {metadata['pair']}.")

        # --- Izlaz iz SHORT pozicije ---
        # Uslov: Pojavljuje se Hammer na 15m sa potvrdom na 5m ili 1m
        # ISPRAVKA: Preimenovano u 'populate_exit_sell' i ispravljen naziv kolone
        exit_short_c1 = dataframe['hammer'] > 0
        exit_short_c2 = (dataframe['hammer_5m'] > 0) | (dataframe['hammer_1m'] > 0)

        dataframe.loc[
            (exit_short_c1 & exit_short_c2),
            'exit_short'] = 1

        if (exit_short_c1 & exit_short_c2).any():
            logger.info(f"✓ Uslov za IZLAZ IZ SHORT pozicije ispunjen za {metadata['pair']}.")

        return dataframe