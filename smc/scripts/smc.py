import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ======= FUNGSI: MOVING AVERAGE (SMA) =======
def sma_rolling(prices, window):
    """Menghitung Simple Moving Average (SMA) dengan jendela tertentu."""
    ser = pd.Series(prices)
    return ser.rolling(window=window).mean().values

# ======= STRATEGI SMC =======
class SmartMoneyConceptStrategy(Strategy):
    """
    Strategi Smart Money Concept (SMC) menggunakan:
    - Break of Structure (BOS)
    - Liquidity Grab
    - Pullback ke area Discount/Premium
    - Entry setelah konfirmasi price action
    """
    n_swing = 10  # Gunakan 10 candle terakhir untuk deteksi swing high/low
    ma_period = 20  # SMA lebih pendek agar lebih responsif
    stoploss_factor = 1.2
    takeprofit_factor = 3.0

    def init(self):
        """Inisialisasi indikator dan variabel bantu"""
        self.ma = self.I(sma_rolling, self.data.Close, self.ma_period)
        self.highs = self.data.High
        self.lows = self.data.Low
        self.last_structure = None  # 'bullish' atau 'bearish'
        self.last_BOS_price = None  # Harga saat BOS terjadi

    def next(self):
        """Logika utama strategi"""
        price = self.data.Close[-1]
        ma_value = self.ma[-1]  # Nilai SMA terbaru

        # ===== DEBUGGING: CETAK HARGA DAN MA =====
        print(f"📈 Price: {price}, SMA: {ma_value}, Last BOS: {self.last_BOS_price}, Structure: {self.last_structure}")

        # ===== DETEKSI SWING HIGH/LOW =====
        if len(self.data) < self.n_swing:
            return

        recent_high = self.data.High.rolling(self.n_swing).max()[-1]
        recent_low = self.data.Low.rolling(self.n_swing).min()[-1]

        # ===== DETEKSI BREAK OF STRUCTURE (BOS) =====
        if price > recent_high:
            if self.last_structure != 'bullish':
                self.last_structure = 'bullish'
                self.last_BOS_price = recent_high
                print(f"🔹 BOS Bullish terjadi di {self.last_BOS_price}")

        elif price < recent_low:
            if self.last_structure != 'bearish':
                self.last_structure = 'bearish'
                self.last_BOS_price = recent_low
                print(f"🔻 BOS Bearish terjadi di {self.last_BOS_price}")

        # ===== ENTRY STRATEGI (BUY / SELL) =====
        if self.last_structure == 'bullish' and price < ma_value * 1.02:
            sl = price - (self.stoploss_factor * (price - self.last_BOS_price))
            tp = price + (self.takeprofit_factor * (price - sl))
            print(f"✅ BUY at {price}, SL: {sl}, TP: {tp}")
            self.buy(sl=sl, tp=tp)

        elif self.last_structure == 'bearish' and price > ma_value * 0.98:
            sl = price + (self.stoploss_factor * (self.last_BOS_price - price))
            tp = price - (self.takeprofit_factor * (sl - price))
            print(f"🔻 SELL at {price}, SL: {sl}, TP: {tp}")
            self.sell(sl=sl, tp=tp)


# ======= FUNGSI BACKTESTING =======
def run_backtest(df):
    """
    Menjalankan backtest dengan dataset pandas DataFrame
    """
    bt = Backtest(
        df,
        SmartMoneyConceptStrategy,
        cash=10_000,
        commission=0.000,  # Sesuaikan dengan broker
        margin=1.0,
        trade_on_close=False,
    )
    stats = bt.run()
    bt.plot()
    return stats


if __name__ == '__main__':
    # ===== LOAD DATA REAL DARI CSV =====
    file_path = r"C:\Users\USER\Documents\xauusd_lstm_project\smc\data\gold_usd_M15.csv"

    df = pd.read_csv(file_path)

    # ===== KONVERSI DATA =====
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Ubah nama kolom agar sesuai dengan Backtesting.py
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }, inplace=True)

    # Pastikan semua kolom bertipe numerik
    df = df.astype(float)

    # ===== DEBUG: CETAK 5 BARIS DATA =====
    print("📊 Contoh Data XAUUSD M15 setelah perubahan format:")
    print(df.head())

    # ===== JALANKAN BACKTEST =====
    stats_result = run_backtest(df)

    # ===== TAMPILKAN HASIL BACKTEST =====
    print(stats_result)
