import MetaTrader5 as mt5

# Inisialisasi MT5
if not mt5.initialize():
    print(f"⚠️ Gagal menghubungkan ke MetaTrader5. Error: {mt5.last_error()}")
    mt5.shutdown()
    exit()

print("✅ MetaTrader5 berhasil terhubung.")
mt5.shutdown()
