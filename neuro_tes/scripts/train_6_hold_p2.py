# Import Library
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ta  # Library untuk indikator teknikal

# Konfigurasi Logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed untuk reproducibility
np.random.seed(42)

# -----------------------------
# 1. Membaca dan Memproses Data
# -----------------------------

data_file = '../data/gold_usd_preprocessed.csv'  # Path CSV preprocessed yang baru

# Membaca data CSV
df = pd.read_csv(data_file, parse_dates=['time'])

# Filter hanya untuk tahun 2024 (sesuai contoh)
df = df[(df['time'] >= '2024-01-01') & (df['time'] <= '2024-12-31')]

# Pastikan data diurutkan berdasarkan waktu
df.sort_values('time', inplace=True)
df.reset_index(drop=True, inplace=True)

# Mengisi nilai yang hilang (jika ada)
df.ffill(inplace=True)

# Definisikan fitur yang akan digunakan dari M15
features = [
    'close_M15', 'RSI_M15', 'EMA_M15', 'MACD_M15',
    'Bollinger_Upper_M15', 'Bollinger_Lower_M15',
    'ADX_M15', 'SMA_M15'
]

# Pastikan semua fitur tersedia
for feature in features:
    if feature not in df.columns:
        raise ValueError(f"Fitur '{feature}' tidak ditemukan dalam data CSV.")

# Ambil fitur yang diperlukan sebagai numpy array
feature_data = df[features].values

print(f"📊 Data setelah pemotongan: {len(df)} baris")
logging.info(f"📊 Data setelah pemotongan: {len(df)} baris")

# -----------------------------
# 2. Definisi Kelas dan Fungsi
# -----------------------------

window_size = 96
initial_money = 50000
population_size = 10
generations = 80
mutation_rate = 0.25
max_trades = 710  # Contoh max_trades kita set lebih kecil untuk melihat efeknya

num_features = len(features)  # Jumlah fitur yang digunakan

class NeuralNetwork:
    def __init__(self, id_, hidden_size=128):
        # Input layer size adalah window_size * num_features
        self.W1 = np.random.randn(window_size * num_features, hidden_size) / np.sqrt(window_size * num_features)
        self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
        self.fitness = 0
        self.id = id_

def relu(X):
    return np.maximum(X, 0)

def softmax(X):
    e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def feed_forward(X, nets):
    a1 = np.dot(X, nets.W1)
    z1 = relu(a1)
    a2 = np.dot(z1, nets.W2)
    return softmax(a2)

# -----------------------------
# 3. Kelas NeuroEvolution
# -----------------------------
class NeuroEvolution:
    def __init__(self, population_size, mutation_rate, model_generator, trend, initial_money, num_features):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_generator = model_generator
        self.trend = trend
        self.initial_money = initial_money
        self.num_features = num_features
        self.population = [self.model_generator(i) for i in range(population_size)]
    
    def mutate(self, individual):
        """Mutasi bobot jaringan dengan probabilitas tertentu."""
        mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W1.shape)
        individual.W1 += np.random.normal(loc=0, scale=0.5, size=individual.W1.shape) * mutation_mask
        mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W2.shape)
        individual.W2 += np.random.normal(loc=0, scale=0.5, size=individual.W2.shape) * mutation_mask
        return individual
    
    def crossover(self, parent1, parent2):
        """Crossover bobot antara dua individu."""
        child = self.model_generator(parent1.id + 1000)
        cutoff = np.random.randint(0, parent1.W1.shape[1])
        child.W1[:, :cutoff] = parent1.W1[:, :cutoff]
        child.W1[:, cutoff:] = parent2.W1[:, cutoff:]
        child.W2[:, :cutoff] = parent1.W2[:, :cutoff]
        child.W2[:, cutoff:] = parent2.W2[:, cutoff:]
        return child
    
    def get_state(self, t):
        """Mengambil jendela data harga terakhir untuk diproses oleh model."""
        if t > 0 and t < window_size:
            window = np.zeros((window_size, self.num_features))
            window[-t:] = self.trend[:t]
        elif t >= window_size:
            window = self.trend[t - window_size:t]
        else:  # t == 0
            window = np.zeros((window_size, self.num_features))
        return window.flatten()
    
    def act(self, individual, state):
        """Menentukan aksi trading berdasarkan output model."""
        # Output model: [P_hold, P_buy, P_sell]
        action = np.argmax(feed_forward(state, individual))
        # Paksa model untuk tidak hanya HOLD (dalam contoh 15% peluang)
        if np.random.rand() < 0.2:
            action = np.random.choice([1, 2])  # Paksa BUY atau SELL
        return action
    
    def evaluate(self, individual):
        """
        Evaluasi fitness dengan fokus utama: Winrate.
        Juga menambahkan penalti jika terlalu banyak HOLD.
        
        Di sini kita juga membagi rata banyaknya trade selama rentang data
        menggunakan bars_between_trades.
        """
        balance = self.initial_money
        total_bars = len(self.trend) - 1
        
        # --------------------------
        # Variabel untuk Drawdown, Winrate, Durasi
        # --------------------------
        peak_equity = balance
        max_drawdown = 0

        closed_trades = 0
        winning_trades = 0
        positions_durations = []
        # --------------------------

        inventory = []
        total_profit = 0
        total_loss = 0

        # Hitung bar jeda minimal antara trade
        bars_between_trades = max(1, total_bars // max_trades)  # agar tidak nol
        last_trade_closed = -bars_between_trades  # boleh langsung trading di awal

        hold_count = 0
        total_trades_done = 0  # total buy+sell

        state = self.get_state(0)
        max_open_positions = 8
        risk_reward_ratio = 1.2
        compound_factor = 0.5

        BARS_PER_DAY = 96
        MAX_BARS_OPEN = 5 * 96

        for t in range(0, total_bars):
            action = self.act(individual, state)
            next_state = self.get_state(t + 1)
            price = self.trend[t][0]  # close_M15

            # Jika action == 0 => HOLD
            if action == 0:
                hold_count += 1

            # Hitung besarnya modal yang mau di-trade
            trade_amount = ((1 + compound_factor) * 0.1) * balance
            atr = np.std([self.trend[i][0] for i in range(max(0, t-96), t)]) if t >= 1 else 0
            stop_loss = price - atr * 3.5
            take_profit = price + (atr * 3.5 * risk_reward_ratio)

            # Membuka posisi baru (jika jeda antar trade sudah cukup, dsb.)
            if (
                len(inventory) < max_open_positions 
                and (t - last_trade_closed) >= bars_between_trades
                and total_trades_done < max_trades  # masih diizinkan
            ):
                if action == 1 and balance >= trade_amount:  # BUY
                    inventory.append({
                        'open_price': price,
                        'invested_amount': trade_amount,
                        'position_type': "BUY",
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'open_t': t
                    })
                    balance -= trade_amount
                    total_trades_done += 1
                    last_trade_closed = t
                elif action == 2 and balance >= trade_amount:  # SELL
                    inventory.append({
                        'open_price': price,
                        'invested_amount': trade_amount,
                        'position_type': "SELL",
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'open_t': t
                    })
                    balance -= trade_amount
                    total_trades_done += 1
                    last_trade_closed = t

            # Hitung Floating Equity untuk Floating Drawdown
            unrealized_PL = 0
            for pos in inventory:
                open_price = pos['open_price']
                invested_amount = pos['invested_amount']
                position_type = pos['position_type']
                if position_type == "BUY":
                    unrealized_PL += ((price - open_price) / open_price) * invested_amount
                else:  # "SELL"
                    unrealized_PL += ((open_price - price) / open_price) * invested_amount

            equity = balance + unrealized_PL
            peak_equity = max(peak_equity, equity)
            if peak_equity != 0:
                current_drawdown = (peak_equity - equity) / peak_equity * 100
            else:
                current_drawdown = 0
            max_drawdown = max(max_drawdown, current_drawdown)

            # Cek Close Posisi (kena SL/TP / max durasi)
            for pos in inventory[:]:
                open_price = pos['open_price']
                invested_amount = pos['invested_amount']
                position_type = pos['position_type']
                open_t = pos['open_t']
                
                duration_in_bars = t - open_t
                price_now = price

                # Apakah harga menembus SL/TP?
                close_position = False
                if duration_in_bars >= MAX_BARS_OPEN:
                    close_position = True
                elif position_type == "BUY":
                    if price_now <= pos['stop_loss'] or price_now >= pos['take_profit']:
                        close_position = True
                else:  # SELL
                    if price_now >= pos['stop_loss'] or price_now <= pos['take_profit']:
                        close_position = True

                if close_position:
                    if position_type == "BUY":
                        profit_trade = ((price_now - open_price) / open_price) * invested_amount
                    else:
                        profit_trade = ((open_price - price_now) / open_price) * invested_amount

                    balance += invested_amount + profit_trade
                    inventory.remove(pos)

                    if profit_trade > 0:
                        winning_trades += 1
                    closed_trades += 1

                    total_profit += max(0, profit_trade)
                    total_loss += abs(min(0, profit_trade))
                    # Tidak kita perbarui last_trade_closed di sini (karena ini penutupan posisi),
                    # agar jeda pembukaan posisi tetap konsisten.

                    positions_durations.append(duration_in_bars / BARS_PER_DAY)

            # Jika saldo habis
            if balance <= 0:
                print(f"⚠️ Individu {individual.id} kehabisan saldo. Menghentikan trading lebih awal.")
                break

            state = next_state

        # Tutup posisi tersisa di akhir
        final_price = self.trend[-1][0]  # close_M15
        final_t = len(self.trend) - 1
        while len(inventory) > 0:
            pos = inventory.pop(0)
            open_price = pos['open_price']
            invested_amount = pos['invested_amount']
            position_type = pos['position_type']

            if position_type == "BUY":
                profit_trade = ((final_price - open_price) / open_price) * invested_amount
            else:
                profit_trade = ((open_price - final_price) / open_price) * invested_amount

            balance += invested_amount + profit_trade

            if profit_trade > 0:
                winning_trades += 1
            closed_trades += 1

            duration_in_bars = final_t - pos['open_t']
            positions_durations.append(duration_in_bars / BARS_PER_DAY)

            total_profit += max(0, profit_trade)
            total_loss += abs(min(0, profit_trade))

            print(f"⚠️ Training berakhir, posisi terakhir ditutup di harga {final_price:.2f} dengan profit {profit_trade:.2f}")

        # ------------------------------------
        # Hitung metrik tambahan
        # ------------------------------------
        profit_dollar = balance - self.initial_money
        profit_percent = (profit_dollar / self.initial_money) * 100

        if closed_trades == 0:
            # Jika tidak ada trade tertutup, anggap saja winrate = 0
            winrate = 0
            avg_duration = 0
            max_duration = 0
        else:
            winrate = (winning_trades / closed_trades) * 100
            avg_duration = sum(positions_durations) / len(positions_durations)
            max_duration = max(positions_durations)

        # Logging ringkas
        logging.info(
            f"Individu {individual.id} - "
            f"Profit: ${profit_dollar:.2f} ({profit_percent:.2f}%), "
            f"Winrate: {winrate:.2f}%, "
            f"Closed Trades: {closed_trades}, "
            f"Max Floating DD: {max_drawdown:.2f}%, "
            f"Avg Duration (days): {avg_duration:.2f}, "
            f"Max Duration (days): {max_duration:.2f}, "
            f"Total Profit: ${total_profit:.2f}, "
            f"Total Loss: ${total_loss:.2f}"
        )

        print(
            f"📊 Individu {individual.id} Summary: "
            f"Profit ${profit_dollar:.2f} ({profit_percent:.2f}%), "
            f"Winrate: {winrate:.2f}%, "
            f"Closed Trades: {closed_trades}, "
            f"Max Floating DD: {max_drawdown:.2f}%, "
            f"Avg Duration (days): {avg_duration:.2f}, "
            f"Max Duration (days): {max_duration:.2f}, "
            f"Total Profit: ${total_profit:.2f}, "
            f"Total Loss: ${total_loss:.2f}"
        )

        # ------------------------------------
        #  Tambahkan penalti terhadap frekuensi HOLD
        # ------------------------------------
        hold_ratio = hold_count / total_bars if total_bars > 0 else 0.0
        fitness_value = winrate - (hold_ratio * 10)  # Contoh penalti terhadap HOLD
        fitness_value = max(0, fitness_value)

        individual.fitness = fitness_value
        return fitness_value


    def evolve(self, generations=50):
        """Proses evolusi dengan memprioritaskan fitness = winrate - penalti_hold."""
        previous_fitness = None

        for epoch in range(generations):
            print(f"\n🔵 Epoch {epoch+1}/{generations} sedang berjalan...")
            for individual in self.population:
                self.evaluate(individual)

            # Sort populasi dari fitness terbesar ke terkecil
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            fittest = self.population[0]

            avg_fitness = np.mean([ind.fitness for ind in self.population])
            worst_fitness = self.population[-1].fitness

            print(f"✅ Epoch {epoch+1} - Fitness Terbaik: {fittest.fitness:.2f}, Rata-rata: {avg_fitness:.2f}, Terburuk: {worst_fitness:.2f}")
            logging.info(f"Epoch {epoch+1}, Fitness Terbaik: {fittest.fitness:.2f}, Rata-rata: {avg_fitness:.2f}, Terburuk: {worst_fitness:.2f}")

            # Jika fitness drop drastis, kurangi mutasi (opsional)
            if epoch > 1 and previous_fitness is not None:
                if fittest.fitness < previous_fitness * 0.5:
                    print("⚠️ Fitness turun drastis, mengurangi mutasi!")
                    logging.info("⚠️ Fitness turun drastis, mengurangi mutasi!")
                    self.mutation_rate *= 0.8

            previous_fitness = fittest.fitness

            model_path = "../models/best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(fittest, f)

            # Membentuk populasi baru
            new_population = [fittest]

            # Crossover + Mutasi (pilih parent dari 10 terbaik)
            for _ in range(self.population_size - 1):
                parent1, parent2 = np.random.choice(self.population[:10], 2)
                child = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child))

            self.population = new_population

        return self.population[0]

# -----------------------------
# 4. Latih Model dan Simpan
# -----------------------------
neural_evolve = NeuroEvolution(population_size, mutation_rate, NeuralNetwork, feature_data, initial_money, num_features)
best_model = neural_evolve.evolve(generations)

print(f"\n🚀 Model terbaik disimpan dengan Fitness (winrate - penalti HOLD) {best_model.fitness:.2f}")

# -----------------------------
# 5. Simulasikan Trading dengan Model Terbaik
#    (Membagi rata trades pada simulasi akhir)
# -----------------------------
def simulate_trading(model, trend, window_size, initial_money, num_features, max_trades=200):
    balance = initial_money
    inventory = []
    buy_points = []
    sell_points = []
    state = None

    total_bars = len(trend) - 1
    bars_between_trades = max(1, total_bars // max_trades)
    last_trade_open = -bars_between_trades

    max_open_positions = 8
    risk_reward_ratio = 1.2
    compound_factor = 0.5
    max_bars_open = 5 * 96  # 5 hari, M15 => 96 bar/hari
    total_trades_done = 0

    BARS_PER_DAY = 96

    balance_series = []  # Opsional: jika ingin memantau per bar
    for t in range(total_bars):
        if t < window_size:
            window = np.zeros((window_size, num_features))
            if t > 0:
                window[-t:] = trend[:t]
        else:
            window = trend[t - window_size:t]
        state = window.flatten()

        # Aksi dari model (P_hold, P_buy, P_sell)
        action = np.argmax(feed_forward(state, model))

        price = trend[t][0]
        atr = np.std([trend[i][0] for i in range(max(0, t-96), t)]) if t >=1 else 0
        trade_amount = ((1 + compound_factor) * 0.1) * balance
        stop_loss = price - atr * 3.5
        take_profit = price + (atr * 3.5 * risk_reward_ratio)

        # Buka posisi baru
        if (
            len(inventory) < max_open_positions
            and (t - last_trade_open) >= bars_between_trades
            and total_trades_done < max_trades
        ):
            if action == 1 and balance >= trade_amount:  # BUY
                inventory.append({
                    'open_price': price,
                    'invested_amount': trade_amount,
                    'position_type': "BUY",
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'open_t': t
                })
                balance -= trade_amount
                buy_points.append(t)
                last_trade_open = t
                total_trades_done += 1
            elif action == 2 and balance >= trade_amount:  # SELL
                inventory.append({
                    'open_price': price,
                    'invested_amount': trade_amount,
                    'position_type': "SELL",
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'open_t': t
                })
                balance -= trade_amount
                sell_points.append(t)
                last_trade_open = t
                total_trades_done += 1

        # Tutup posisi jika SL/TP atau durasi maksimum
        for pos in inventory[:]:
            open_price = pos['open_price']
            invested_amount = pos['invested_amount']
            position_type = pos['position_type']
            open_t = pos['open_t']

            duration_in_bars = t - open_t
            price_now = price

            # Update trailing stop (opsional, bisa ditambahkan)
            # ...

            close_position = False
            if duration_in_bars >= max_bars_open:
                close_position = True
            elif position_type == "BUY":
                # Kena SL/TP
                if price_now <= pos['stop_loss'] or price_now >= pos['take_profit']:
                    close_position = True
            else:  # SELL
                if price_now >= pos['stop_loss'] or price_now <= pos['take_profit']:
                    close_position = True

            if close_position:
                if position_type == "BUY":
                    profit_trade = ((price_now - open_price) / open_price) * invested_amount
                else:
                    profit_trade = ((open_price - price_now) / open_price) * invested_amount

                balance += invested_amount + profit_trade
                inventory.remove(pos)

        balance_series.append(balance)

        # Jika saldo habis
        if balance <= 0:
            print("⚠️ Saldo habis. Menghentikan trading lebih awal.")
            break

        # Batas total trades tercapai
        if total_trades_done >= max_trades:
            print(f"⚠️ Mencapai batas {max_trades} transaksi. Menghentikan trading!")
            break

    # Tutup semua posisi tersisa di akhir
    final_price = trend[-1][0]
    final_t = len(trend) - 1
    while len(inventory) > 0:
        pos = inventory.pop(0)
        open_price = pos['open_price']
        invested_amount = pos['invested_amount']
        position_type = pos['position_type']

        if position_type == "BUY":
            profit_trade = ((final_price - open_price) / open_price) * invested_amount
        else:
            profit_trade = ((open_price - final_price) / open_price) * invested_amount

        balance += invested_amount + profit_trade

    return buy_points, sell_points, balance_series

# Simulasikan trading dengan model terbaik (trading terdistribusi)
buy_points, sell_points, balance_series = simulate_trading(
    best_model, feature_data, window_size, initial_money, num_features, max_trades
)

# -----------------------------
# 6. Identifikasi Zona Supply dan Demand (Opsional)
# -----------------------------
def identify_supply_demand(df, window=50, threshold=0.02):
    """Identifikasi zona supply dan demand berdasarkan pivot points sederhana."""
    supply_zones = []
    demand_zones = []
    
    for i in range(window, len(df)-window):
        window_data = df['close_M15'][i-window:i+window]
        current = df['close_M15'][i]
        high = window_data.max()
        low = window_data.min()
        
        # Jika current adalah high lokal
        if current == high and (high - low) > threshold * low:
            supply_zones.append((df['time'][i], high))
        
        # Jika current adalah low lokal
        if current == low and (high - low) > threshold * low:
            demand_zones.append((df['time'][i], low))
    
    return supply_zones, demand_zones

supply_zones, demand_zones = identify_supply_demand(df)

# -----------------------------
# 7. Plot Hasil Trading dan Supply/Demand (Opsional)
# -----------------------------
def plot_trading(df, buy_points, sell_points, supply_zones, demand_zones):
    plt.figure(figsize=(15, 7))
    plt.plot(df['time'], df['close_M15'], label='Close Price', color='blue')

    # Plot Buy Points
    if buy_points:
        plt.scatter(df['time'].iloc[buy_points], df['close_M15'].iloc[buy_points], 
                    marker='^', color='green', label='Buy', s=100)

    # Plot Sell Points
    if sell_points:
        plt.scatter(df['time'].iloc[sell_points], df['close_M15'].iloc[sell_points], 
                    marker='v', color='red', label='Sell', s=100)

    # Plot Supply Zones
    for zone in supply_zones:
        plt.axhline(y=zone[1], color='magenta', linestyle='--', linewidth=0.8)
        plt.text(zone[0], zone[1], 'Supply', color='magenta', fontsize=8, verticalalignment='bottom')

    # Plot Demand Zones
    for zone in demand_zones:
        plt.axhline(y=zone[1], color='cyan', linestyle='--', linewidth=0.8)
        plt.text(zone[0], zone[1], 'Demand', color='cyan', fontsize=8, verticalalignment='top')

    plt.title('Hasil Trading (Distribusi Merata & Penalti HOLD)')
    plt.xlabel('Waktu')
    plt.ylabel('Harga Close M15')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_trading(df, buy_points, sell_points, supply_zones, demand_zones)
