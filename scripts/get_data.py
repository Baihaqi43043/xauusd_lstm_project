import MetaTrader5 as mt5
import pandas as pd
import datetime

# Hubungkan ke MetaTrader5
mt5.initialize()

# Simbol XAU/USD (pastikan simbolnya benar)
symbol = "XAUUSD"

# Timeframes yang dibutuhkan
timeframes = {
    "M1": mt5.TIMEFRAME_M1,   # Timeframe 1 menit
    "M5": mt5.TIMEFRAME_M5,   # Timeframe 5 menit
    "M15": mt5.TIMEFRAME_M15  # Timeframe 15 menit
}

# Rentang waktu (ambil data 6 bulan terakhir)
start = datetime.datetime.now() - datetime.timedelta(days=365 * 5)
end = datetime.datetime.now()

# Loop untuk mengambil data dari timeframe yang berbeda
for tf_name, tf in timeframes.items():
    rates = mt5.copy_rates_range(symbol, tf, start, end)
    df = pd.DataFrame(rates)

    # Jika data kosong, beri tahu user
    if df.empty:
        print(f"⚠️ Tidak ada data untuk timeframe {tf_name}")
        continue

    # Konversi Unix timestamp ke datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Simpan ke CSV
    csv_filename = f"gold_usd_{tf_name}.csv"
    df.to_csv(csv_filename)

    print(f"✅ Data {tf_name} berhasil diunduh dan disimpan ke {csv_filename}")

mt5.shutdown()
