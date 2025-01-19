import MetaTrader5 as mt5

# Hubungkan ke MetaTrader5
mt5.initialize()

# Ambil daftar semua simbol yang tersedia di broker
symbols = mt5.symbols_get()

# Cetak semua simbol
print("✅ Semua simbol yang tersedia di broker:")
for symbol in symbols:
    print(symbol.name)

mt5.shutdown()
