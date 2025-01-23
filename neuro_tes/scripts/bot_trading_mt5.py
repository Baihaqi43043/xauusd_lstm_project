# Import Library
import MetaTrader5 as mt5
import numpy as np
import pickle
import time
import logging

# Konfigurasi Logging
logging.basicConfig(filename='trading.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Parameter Akun MT5 (Ganti dengan akun Anda)
MT5_LOGIN = 123456789  # Ganti dengan Login Akun MT5 Anda
MT5_PASSWORD = "password_anda"  # Ganti dengan Password Akun MT5 Anda
MT5_SERVER = "Broker-Server"  # Ganti dengan Nama Server Broker Anda

# Parameter Trading
SYMBOL = "XAUUSD"
LOT_SIZE = 0.1
SL_PERCENT = 0.5  # Stop Loss dalam persen
TP_PERCENT = 1.0  # Take Profit dalam persen
WINDOW_SIZE = 96  # Window Size yang digunakan model
TRADE_INTERVAL = 60 * 60  # Waktu antara trading dalam detik (60 detik = 1 menit)

# -----------------------------
# 1. Fungsi untuk Koneksi ke MT5
# -----------------------------

def connect_mt5():
    """Menghubungkan ke akun MetaTrader 5"""
    if not mt5.initialize():
        print("❌ Gagal menghubungkan ke MetaTrader 5")
        logging.error("Gagal menghubungkan ke MetaTrader 5")
        return False

    # Login ke akun trading
    authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        print(f"❌ Gagal login ke akun MT5! Error: {mt5.last_error()}")
        logging.error(f"Gagal login ke akun MT5! Error: {mt5.last_error()}")
        return False

    print(f"✅ Berhasil login ke akun MT5: {MT5_LOGIN} di server {MT5_SERVER}")
    logging.info(f"Berhasil login ke akun MT5: {MT5_LOGIN} di server {MT5_SERVER}")
    return True

# -----------------------------
# 2. Fungsi untuk Mengambil Harga Real-time
# -----------------------------

def get_latest_prices(symbol, count=WINDOW_SIZE):
    """Mengambil harga terbaru dari MetaTrader 5"""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, count)
    if rates is None:
        logging.error("Gagal mendapatkan harga dari MT5")
        return None
    return np.array([rate['close'] for rate in rates])

# -----------------------------
# 3. Fungsi untuk Menghitung Stop Loss (SL) dan Take Profit (TP)
# -----------------------------

def calculate_sl_tp(entry_price, sl_percent, tp_percent, trade_type):
    """Menghitung Stop Loss dan Take Profit berdasarkan persentase harga"""
    if trade_type == "BUY":
        stop_loss = entry_price * (1 - sl_percent / 100)
        take_profit = entry_price * (1 + tp_percent / 100)
    else:  # SELL
        stop_loss = entry_price * (1 + sl_percent / 100)
        take_profit = entry_price * (1 - tp_percent / 100)
    return stop_loss, take_profit

# -----------------------------
# 4. Fungsi untuk Mengirim Order ke MT5
# -----------------------------

def place_order(symbol, trade_type, volume, sl, tp):
    """Mengirim order BUY atau SELL ke MetaTrader 5"""
    price = mt5.symbol_info_tick(symbol).ask if trade_type == "BUY" else mt5.symbol_info_tick(symbol).bid
    order_type_mt5 = mt5.ORDER_BUY if trade_type == "BUY" else mt5.ORDER_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type_mt5,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "AI Trading Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order gagal: {result.retcode}")
        print(f"Order gagal: {result.retcode}")
    else:
        logging.info(f"Order sukses: {trade_type} @ {price}, SL: {sl}, TP: {tp}")
        print(f"Order sukses: {trade_type} @ {price}, SL: {sl}, TP: {tp}")

# -----------------------------
# 5. Fungsi untuk Memuat Model
# -----------------------------

def load_model(model_path):
    """Memuat model terbaik dari file"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# -----------------------------
# 6. Fungsi untuk Prediksi dan Eksekusi Trading
# -----------------------------

def trade(symbol, model):
    """Mengambil keputusan trading berdasarkan model"""
    prices = get_latest_prices(symbol)
    if prices is None or len(prices) < WINDOW_SIZE:
        print("Data harga tidak cukup, menunggu data baru...")
        return

    # Konversi harga menjadi input untuk model
    state = np.array([prices[-WINDOW_SIZE:]])

    # Prediksi tindakan (0 = Hold, 1 = Buy, 2 = Sell)
    action = np.argmax(feed_forward(state, model))

    if action == 1:  # BUY
        price = mt5.symbol_info_tick(symbol).ask
        sl, tp = calculate_sl_tp(price, SL_PERCENT, TP_PERCENT, "BUY")
        place_order(symbol, "BUY", LOT_SIZE, sl, tp)

    elif action == 2:  # SELL
        price = mt5.symbol_info_tick(symbol).bid
        sl, tp = calculate_sl_tp(price, SL_PERCENT, TP_PERCENT, "SELL")
        place_order(symbol, "SELL", LOT_SIZE, sl, tp)

# -----------------------------
# 7. Fungsi untuk Menjalankan Trading Otomatis
# -----------------------------

def run_bot():
    """Menjalankan bot trading otomatis"""
    if not connect_mt5():
        return

    model_path = "models/best_model.pkl"
    model = load_model(model_path)
    
    print("Bot Trading XAUUSD Dimulai...")
    logging.info("Bot Trading XAUUSD Dimulai...")

    try:
        while True:
            trade(SYMBOL, model)
            print(f"Menunggu {TRADE_INTERVAL} detik sebelum trading berikutnya...")
            time.sleep(TRADE_INTERVAL)
    except KeyboardInterrupt:
        print("Bot Trading Dihentikan.")
        logging.info("Bot Trading Dihentikan.")
        mt5.shutdown()

# -----------------------------
# 8. Jalankan Bot Trading
# -----------------------------

if __name__ == "__main__":
    run_bot()
