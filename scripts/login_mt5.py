import MetaTrader5 as mt5

# Ganti dengan informasi akun MT5 kamu
ACCOUNT = 48687812  # Nomor akun trading
PASSWORD = "Haloboy12!"  # Password akun
SERVER = "HFMarketsGlobal-Demo"  # Server broker

# Hubungkan ke MetaTrader5
if not mt5.initialize():
    print("⚠️ Gagal menghubungkan ke MetaTrader5.")
    mt5.shutdown()
    exit()

# Login ke akun
if mt5.login(ACCOUNT, PASSWORD, SERVER):
    print(f"✅ Berhasil login ke akun: {ACCOUNT}")
else:
    print(f"⚠️ Gagal login ke akun {ACCOUNT}. Periksa nomor akun, password, dan server.")

mt5.shutdown()
