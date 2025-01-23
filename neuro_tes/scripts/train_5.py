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
# 1. Membaca dan Memproses Data (Diperbarui)
# -----------------------------

data_file = '../data/gold_usd_preprocessed.csv'  # Path CSV preprocessed yang baru

# Membaca data CSV
df = pd.read_csv(data_file, parse_dates=['time'])

# Filter hanya untuk tahun 2024
df = df[(df['time'] >= '2024-01-01') & (df['time'] <= '2024-12-31')]

# Pastikan data diurutkan berdasarkan waktu
df.sort_values('time', inplace=True)
df.reset_index(drop=True, inplace=True)

# Mengisi nilai yang hilang
df.ffill(inplace=True)

# Definisikan fitur yang akan digunakan dari M15
features = [
    'close_M15', 'RSI_M15', 'EMA_M15', 'MACD_M15', 
    'Bollinger_Upper_M15', 'Bollinger_Lower_M15'
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
population_size = 2
generations = 120
mutation_rate = 0.2

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
        """Mutasi bobot jaringan dengan probabilitas lebih tinggi"""
        mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W1.shape)
        individual.W1 += np.random.normal(loc=0, scale=0.5, size=individual.W1.shape) * mutation_mask
        mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W2.shape)
        individual.W2 += np.random.normal(loc=0, scale=0.5, size=individual.W2.shape) * mutation_mask
        return individual
    
    def crossover(self, parent1, parent2):
        """Crossover bobot antara dua individu"""
        child = self.model_generator(parent1.id + 1000)
        cutoff = np.random.randint(0, parent1.W1.shape[1])
        child.W1[:, :cutoff] = parent1.W1[:, :cutoff]
        child.W1[:, cutoff:] = parent2.W1[:, cutoff:]
        child.W2[:, :cutoff] = parent1.W2[:, :cutoff]
        child.W2[:, cutoff:] = parent2.W2[:, cutoff:]
        return child
    
    def get_state(self, t):
        """Mengambil jendela data harga terakhir untuk diproses oleh model"""
        if t > 0 and t < window_size:
            window = np.zeros((window_size, self.num_features))
            window[-t:] = self.trend[:t]
        elif t >= window_size:
            window = self.trend[t - window_size:t]
        else:  # t == 0
            window = np.zeros((window_size, self.num_features))
        return window.flatten()
    
    def act(self, individual, state):
        """Menentukan aksi trading berdasarkan model"""
        action = np.argmax(feed_forward(state, individual))

        # Paksa model untuk tidak hanya HOLD
        if np.random.rand() < 0.75:
            action = np.random.choice([1, 2])  # Paksa sesekali BUY atau SELL

        return action
    
    def evaluate(self, individual):
        """Evaluasi fitness berdasarkan total profit dari trading"""
        balance = self.initial_money

        # --------------------------
        # Variabel untuk Drawdown, Winrate, Durasi
        # --------------------------
        peak_equity = balance
        max_drawdown = 0

        closed_trades = 0
        winning_trades = 0
        positions_durations = []  # Menyimpan lama (dalam hari) tiap posisi yang ditutup
        # --------------------------

        inventory = []
        total_trades = 0
        total_profit = 0
        total_loss = 0
        state = self.get_state(0)
        cooldown_period = 12
        last_trade_closed = -cooldown_period  # Initialize to allow immediate trading

        max_trades = 10000
        max_open_positions = 5
        risk_reward_ratio = 5
        compound_factor = 0.5

        # Constants for duration in bars
        BARS_PER_DAY = 96         # M15 => 96 bar = 1 hari
        MAX_BARS_OPEN = 5 * 96    # 5 hari = 480 bar

        for t in range(0, len(self.trend) - 1):
            action = self.act(individual, state)
            next_state = self.get_state(t + 1)
            price = self.trend[t][0]  # close_M15

            trade_amount = ((1 + compound_factor) * 0.125) * balance  # Menggunakan 1.5% dari saldo untuk setiap trade

            atr = np.std([self.trend[i][0] for i in range(max(0, t-96), t)]) if t >=1 else 0
            stop_loss = price - atr * 3.5
            take_profit = price + (atr * 3.5 * risk_reward_ratio)
            trailing_stop = stop_loss  # Initialize trailing stop

            # Membuka posisi baru (jika diizinkan)
            if len(inventory) < max_open_positions and (t - last_trade_closed) >= cooldown_period:
                if action == 1 and balance >= trade_amount:  # BUY
                    # Tambahkan open_t = t dan trailing_stop
                    inventory.append({
                        'open_price': price,
                        'invested_amount': trade_amount,
                        'position_type': "BUY",
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'open_t': t,
                        'trailing_stop': trailing_stop
                    })
                    balance -= trade_amount
                    total_trades += 1
                elif action == 2 and balance >= trade_amount:  # SELL
                    # Tambahkan open_t = t dan trailing_stop
                    inventory.append({
                        'open_price': price,
                        'invested_amount': trade_amount,
                        'position_type': "SELL",
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'open_t': t,
                        'trailing_stop': trailing_stop
                    })
                    balance -= trade_amount
                    total_trades += 1

            # ---- Hitung Floating Equity untuk Floating Drawdown ----
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
            current_drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity !=0 else 0
            max_drawdown = max(max_drawdown, current_drawdown)

            # ---- Cek Close Posisi (kena SL/TP / max durasi / trailing stop) ----
            for pos in inventory[:]:
                open_price = pos['open_price']
                invested_amount = pos['invested_amount']
                position_type = pos['position_type']
                sl = pos['stop_loss']
                tp = pos['take_profit']
                open_t = pos['open_t']
                trailing_stop = pos['trailing_stop']

                duration_in_bars = t - open_t
                price_now = price

                # Update trailing stop
                if position_type == "BUY":
                    if price_now > pos['open_price']:
                        new_trailing_stop = price_now - atr * 3.5
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_trailing_stop)
                else:  # SELL
                    if price_now < pos['open_price']:
                        new_trailing_stop = price_now + atr * 3.5
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_trailing_stop)

                # Kondisi untuk menutup posisi
                close_position = False
                if duration_in_bars >= MAX_BARS_OPEN:
                    close_position = True
                elif position_type == "BUY":
                    if price_now <= pos['trailing_stop'] or price_now >= tp:
                        close_position = True
                else:  # SELL
                    if price_now >= pos['trailing_stop'] or price_now <= tp:
                        close_position = True

                if close_position:
                    if position_type == "BUY":
                        profit_trade = ((price_now - open_price) / open_price) * invested_amount
                    else:
                        profit_trade = ((open_price - price_now) / open_price) * invested_amount

                    balance += invested_amount + profit_trade
                    inventory.remove(pos)
                    total_profit += max(0, profit_trade)
                    total_loss += abs(min(0, profit_trade))

                    closed_trades += 1
                    if profit_trade > 0:
                        winning_trades += 1

                    # Simpan durasi (hari)
                    positions_durations.append(duration_in_bars / BARS_PER_DAY)

                    total_trades += 1
                    last_trade_closed = t

            # Batasan total transaksi
            if total_trades >= max_trades:
                print(f"⚠️ Individu {individual.id} mencapai batas {max_trades} transaksi. Menghentikan trading!")
                break

            # Jika saldo habis
            if balance <= 0:
                print(f"⚠️ Individu {individual.id} kehabisan saldo. Menghentikan trading lebih awal.")
                break

            state = next_state

        # ---- Tutup posisi tersisa di akhir ----
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
            total_profit += max(0, profit_trade)
            total_loss += abs(min(0, profit_trade))

            closed_trades += 1
            if profit_trade > 0:
                winning_trades += 1

            # Hitung durasi sisa (hari)
            duration_in_bars = final_t - pos['open_t']
            positions_durations.append(duration_in_bars / BARS_PER_DAY)

            total_trades += 1
            print(f"⚠️ Training berakhir, posisi terakhir ditutup di harga {final_price:.2f} dengan profit {profit_trade:.2f}")

        # ---- Hitung profit (dollar dan %) ----
        profit_dollar = balance - self.initial_money
        profit_percent = (profit_dollar / self.initial_money) * 100

        # ---- Hitung Winrate, Avg Duration (hari), Max Duration (hari) ----
        if closed_trades == 0:
            profit_percent = -1000
            profit_dollar = -self.initial_money
            winrate = 0
            avg_duration = 0
            max_duration = 0
        else:
            winrate = (winning_trades / closed_trades) * 100
            avg_duration = sum(positions_durations) / len(positions_durations)  # dalam hari
            max_duration = max(positions_durations)  # dalam hari

        logging.info(
            f"Individu {individual.id} - "
            f"Profit: ${profit_dollar:.2f} ({profit_percent:.2f}%), "
            f"Total Trades: {total_trades}, "
            f"Closed Trades: {closed_trades}, "
            f"Winrate: {winrate:.2f}%, "
            f"Max Floating DD: {max_drawdown:.2f}%, "
            f"Avg Duration (days): {avg_duration:.2f}, "
            f"Max Duration (days): {max_duration:.2f}, "
            f"Total Profit: ${total_profit:.2f}, "
            f"Total Loss: ${total_loss:.2f}"
        )

        print(
            f"📊 Individu {individual.id} Summary: "
            f"Profit ${profit_dollar:.2f} ({profit_percent:.2f}%), "
            f"Closed Trades: {closed_trades}, "
            f"Winrate: {winrate:.2f}%, "
            f"Max Floating DD: {max_drawdown:.2f}%, "
            f"Avg Duration (days): {avg_duration:.2f}, "
            f"Max Duration (days): {max_duration:.2f}, "
            f"Total Profit: ${total_profit:.2f}, "
            f"Total Loss: ${total_loss:.2f}"
        )

        individual.fitness = profit_percent
        return profit_percent


    def evolve(self, generations=50):
        """Proses evolusi untuk menemukan model terbaik"""
        previous_fitness = None

        for epoch in range(generations):
            print(f"\n🔵 Epoch {epoch+1}/{generations} sedang berjalan...")
            for individual in self.population:
                self.evaluate(individual)

            self.population.sort(key=lambda x: x.fitness, reverse=True)
            fittest = self.population[0]

            avg_fitness = np.mean([ind.fitness for ind in self.population])
            worst_fitness = self.population[-1].fitness

            print(f"✅ Epoch {epoch+1} - Fitness Terbaik: {fittest.fitness:.2f}, Rata-rata: {avg_fitness:.2f}, Terburuk: {worst_fitness:.2f}")
            logging.info(f"Epoch {epoch+1}, Fitness Terbaik: {fittest.fitness:.2f}, Rata-rata: {avg_fitness:.2f}, Terburuk: {worst_fitness:.2f}")

            if epoch > 1 and previous_fitness is not None:
                if fittest.fitness < previous_fitness * 0.5:
                    print("⚠️ Fitness turun drastis, mengurangi mutasi!")
                    logging.info("⚠️ Fitness turun drastis, mengurangi mutasi!")
                    self.mutation_rate *= 0.8

            previous_fitness = fittest.fitness

            model_path = "../models/best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(fittest, f)

            new_population = [fittest]

            for _ in range(self.population_size - 1):
                parent1, parent2 = np.random.choice(self.population[:10], 2)
                child = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child))

            self.population = new_population

        return self.population[0]

# -----------------------------
# 4. Latih Model dan Simpan ke File (Diperbarui)
# -----------------------------

neural_evolve = NeuroEvolution(population_size, mutation_rate, NeuralNetwork, feature_data, initial_money, num_features)
best_model = neural_evolve.evolve(generations)

print(f"\n🚀 Model terbaik disimpan dengan fitness {best_model.fitness:.2f}")
logging.info(f"Model terbaik disimpan dengan fitness {best_model.fitness:.2f}")

# -----------------------------
# 5. Simulasikan Trading dengan Model Terbaik
# -----------------------------

def simulate_trading(model, trend, window_size, initial_money, num_features):
    balance = initial_money
    inventory = []
    buy_points = []
    sell_points = []
    state = None
    cooldown_period = 8  # Definisikan cooldown_period sebelum digunakan
    last_trade_closed = -cooldown_period  # Initialize to allow immediate trading
    BARS_PER_DAY = 96
    max_open_positions = 5
    max_trades = 10000
    compound_factor = 0.05
    risk_reward_ratio = 5
    max_bars_open = 5 * BARS_PER_DAY

    for t in range(len(trend) - 1):
        if t < window_size:
            window = np.zeros((window_size, num_features))
            if t > 0:
                window[-t:] = trend[:t]
        else:
            window = trend[t - window_size:t]
        state = window.flatten()

        action = np.argmax(feed_forward(state, model))

        # Paksa model untuk tidak hanya HOLD
        if np.random.rand() < 0.1:
            action = np.random.choice([1, 2])  # BUY atau SELL

        price = trend[t][0]  # close_M15
        atr = np.std([trend[i][0] for i in range(max(0, t-96), t)]) if t >=1 else 0
        trade_amount = ((1 + compound_factor) * 0.15) * balance  # Menggunakan 1.5% dari saldo untuk setiap trade
        stop_loss = price - atr * 3.5
        take_profit = price + (atr * 3.5 * risk_reward_ratio)
        trailing_stop = stop_loss  # Initialize trailing stop

        # Buka posisi baru
        if len(inventory) < max_open_positions and (t - last_trade_closed) >= cooldown_period:
            if action == 1 and balance >= trade_amount:  # BUY
                inventory.append({
                    'open_price': price,
                    'invested_amount': trade_amount,
                    'position_type': "BUY",
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'open_t': t,
                    'trailing_stop': trailing_stop
                })
                balance -= trade_amount
                buy_points.append(t)
            elif action == 2 and balance >= trade_amount:  # SELL
                inventory.append({
                    'open_price': price,
                    'invested_amount': trade_amount,
                    'position_type': "SELL",
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'open_t': t,
                    'trailing_stop': trailing_stop
                })
                balance -= trade_amount
                sell_points.append(t)

        # Tutup posisi jika mencapai SL/TP atau durasi maksimum atau trailing stop
        for pos in inventory[:]:
            open_price = pos['open_price']
            invested_amount = pos['invested_amount']
            position_type = pos['position_type']
            sl = pos['stop_loss']
            tp = pos['take_profit']
            open_t = pos['open_t']
            trailing_stop = pos['trailing_stop']

            duration_in_bars = t - open_t
            price_now = price

            # Update trailing stop
            if position_type == "BUY":
                if price_now > pos['open_price']:
                    new_trailing_stop = price_now - atr * 3.5
                    pos['trailing_stop'] = max(pos['trailing_stop'], new_trailing_stop)
            else:  # SELL
                if price_now < pos['open_price']:
                    new_trailing_stop = price_now + atr * 3.5
                    pos['trailing_stop'] = min(pos['trailing_stop'], new_trailing_stop)

            # Kondisi untuk menutup posisi
            close_position = False
            if duration_in_bars >= max_bars_open:
                close_position = True
            elif position_type == "BUY":
                if price_now <= pos['trailing_stop'] or price_now >= tp:
                    close_position = True
            else:  # SELL
                if price_now >= pos['trailing_stop'] or price_now <= tp:
                    close_position = True

            if close_position:
                if position_type == "BUY":
                    profit_trade = ((price_now - open_price) / open_price) * invested_amount
                else:
                    profit_trade = ((open_price - price_now) / open_price) * invested_amount

                balance += invested_amount + profit_trade
                inventory.remove(pos)
                if position_type == "BUY":
                    if profit_trade > 0:
                        sell_points.append(t)
                else:
                    if profit_trade > 0:
                        buy_points.append(t)

        # Batasan total transaksi
        if len(buy_points) + len(sell_points) >= max_trades:
            print(f"⚠️ Mencapai batas {max_trades} transaksi. Menghentikan trading!")
            break

        # Jika saldo habis
        if balance <= 0:
            print("⚠️ Saldo habis. Menghentikan trading lebih awal.")
            break

    # Tutup posisi yang masih terbuka di akhir
    final_price = trend[-1][0]  # close_M15
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

        if position_type == "BUY":
            sell_points.append(final_t)
        else:
            buy_points.append(final_t)

    return buy_points, sell_points

# Simulasikan trading dengan model terbaik
buy_points, sell_points = simulate_trading(best_model, feature_data, window_size, initial_money, num_features)

# -----------------------------
# 6. Identifikasi Zona Supply dan Demand
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
# 7. Plot Hasil Trading dan Indikator Supply/Demand
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

    plt.title('Hasil Trading dengan Model Terbaik')
    plt.xlabel('Waktu')
    plt.ylabel('Harga Close M15')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Panggil fungsi plot
plot_trading(df, buy_points, sell_points, supply_zones, demand_zones)
