# ======================================================
# 1. Import Library & Konfigurasi
# ======================================================
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Konfigurasi Logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed untuk reproducibility
np.random.seed(42)

# ======================================================
# 2. Membaca & Memproses Data
# ======================================================
data_file = '../data/gold_usd_preprocessed.csv'  # Ganti path dengan CSV Anda
df = pd.read_csv(data_file, parse_dates=['time'])

# Filter hanya untuk tahun 2024 (Contoh)
df = df[(df['time'] >= '2024-01-01') & (df['time'] <= '2024-12-31')]
df.sort_values('time', inplace=True)
df.reset_index(drop=True, inplace=True)

# Mengisi nilai yang hilang (jika ada)
df.ffill(inplace=True)

# Definisikan fitur yang akan digunakan dari M15
features = ['close_M15', 'RSI_M15', 'EMA_M15', 'MACD_M15', 
            'Bollinger_Upper_M15', 'Bollinger_Lower_M15']

# Cek apakah fitur tersedia
for feature in features:
    if feature not in df.columns:
        raise ValueError(f"Fitur '{feature}' tidak ditemukan dalam data CSV.")

# Ambil data fitur
feature_data = df[features].values

print(f"📊 Data setelah pemotongan: {len(df)} baris")
logging.info(f"📊 Data setelah pemotongan: {len(df)} baris")

# ======================================================
# 3. Fungsi Identifikasi Zona Supply/Demand
# ======================================================
def identify_supply_demand(df, window=50, threshold=0.01, min_frequency=3):
    """
    Identifikasi zona supply/demand berdasarkan pivot sederhana 
    dengan frekuensi minimum tertentu.
    
    window     : lebar window di kiri-kanan bar
    threshold  : persentase (range high-low minimal)
    min_frequency : berapa kali level high/low muncul agar jadi zona kuat
    """
    supply_zones = {}
    demand_zones = {}
    
    for i in range(window, len(df)-window):
        window_data = df['close_M15'][i-window:i+window]
        current = df['close_M15'][i]
        high = window_data.max()
        low = window_data.min()
        
        # range minimal = threshold*low
        # jika current adalah local high
        if current == high and (high - low) > threshold * low:
            supply_zones[high] = supply_zones.get(high, 0) + 1
        
        # jika current adalah local low
        if current == low and (high - low) > threshold * low:
            demand_zones[low] = demand_zones.get(low, 0) + 1
    
    # Filter zona berdasarkan frekuensi minimum
    strong_supply_zones = [(price, freq) for price, freq in supply_zones.items() if freq >= min_frequency]
    strong_demand_zones = [(price, freq) for price, freq in demand_zones.items() if freq >= min_frequency]
    
    return strong_supply_zones, strong_demand_zones

# Identifikasi zona supply dan demand
supply_zones, demand_zones = identify_supply_demand(df)

# ======================================================
# 4. Parameter Awal dan Kelas NeuralNetwork
# ======================================================
window_size = 96     # jumlah bar (M15) yang dijadikan 1 input window
initial_money = 50000
population_size = 20
generations = 5     # Dikurangi sekadar contoh, silakan naikkan kembali
mutation_rate = 0.2

num_features = len(features)

class NeuralNetwork:
    def __init__(self, id_, hidden_size=128):
        # Bobot acak normal disesuaikan dengan sqrt input size
        self.W1 = np.random.randn(window_size * num_features, hidden_size) / np.sqrt(window_size * num_features)
        self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
        self.fitness = 0
        self.id = id_

# ======================================================
# 5. Fungsi Aktivasi & Feed Forward
# ======================================================
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

# ======================================================
# 6. Kelas NeuroEvolution
# ======================================================
class NeuroEvolution:
    def __init__(self, population_size, mutation_rate, model_generator, trend, initial_money, num_features, supply_zones, demand_zones):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_generator = model_generator
        self.trend = trend  # array of shape (N, num_features)
        self.initial_money = initial_money
        self.num_features = num_features
        
        # Hanya ambil harga (price) dari supply_zones, demand_zones
        self.supply_zones = [zone[0] for zone in supply_zones]
        self.demand_zones = [zone[0] for zone in demand_zones]
        
        # (Opsional) Definisikan liquidity_zones secara statis/placeholder
        self.liquidity_zones = [1920, 1930, 1950]  # contoh fiksi
        
        # Inisialisasi populasi neural network
        self.population = [self.model_generator(i) for i in range(population_size)]

    # ================ Fungsi Bantuan Zona ================
    def is_in_supply_zone(self, price, tolerance=0.01):
        """
        Return True jika harga berada di sekitar supply zone (±tolerance).
        """
        for zone in self.supply_zones:
            if abs(price - zone) / zone <= tolerance:
                return True
        return False
    
    def is_in_demand_zone(self, price, tolerance=0.01):
        """
        Return True jika harga berada di sekitar demand zone (±tolerance).
        """
        for zone in self.demand_zones:
            if abs(price - zone) / zone <= tolerance:
                return True
        return False

    def is_in_liquidity_zone(self, price, tolerance=0.01):
        """
        Return True jika price berada di sekitar salah satu liquidity_zones.
        """
        for zone in self.liquidity_zones:
            if abs(price - zone) / zone <= tolerance:
                return True
        return False

    # ================ Fungsi GA: Mutasi & Crossover ================
    def mutate(self, individual):
        """Mutasi bobot jaringan dengan probabilitas self.mutation_rate."""
        # Mutasi W1
        mutation_mask_W1 = np.random.binomial(1, p=self.mutation_rate, size=individual.W1.shape)
        individual.W1 += np.random.normal(loc=0, scale=0.5, size=individual.W1.shape) * mutation_mask_W1

        # Mutasi W2
        mutation_mask_W2 = np.random.binomial(1, p=self.mutation_rate, size=individual.W2.shape)
        individual.W2 += np.random.normal(loc=0, scale=0.5, size=individual.W2.shape) * mutation_mask_W2

        return individual
    
    def crossover(self, parent1, parent2):
        """
        Crossover bobot sederhana.
        """
        child = self.model_generator(parent1.id + 1000)
        cutoff = np.random.randint(0, parent1.W1.shape[1])
        # Crossover W1
        child.W1[:, :cutoff] = parent1.W1[:, :cutoff]
        child.W1[:, cutoff:] = parent2.W1[:, cutoff:]
        # Crossover W2
        child.W2[:, :cutoff] = parent1.W2[:, :cutoff]
        child.W2[:, cutoff:] = parent2.W2[:, cutoff:]
        return child
    
    # ================ Fungsi State ================
    def get_state(self, t):
        """
        Mengambil window data terakhir (size=window_size)
        untuk input ke neural network.
        """
        if t > 0 and t < window_size:
            window = np.zeros((window_size, self.num_features))
            window[-t:] = self.trend[:t]
        elif t >= window_size:
            window = self.trend[t - window_size:t]
        else:  # t == 0
            window = np.zeros((window_size, self.num_features))
        return window.flatten()

    # ================ Fungsi ACT: Kombinasi NN + Zona =================
    def act(self, individual, state, current_price):
        """
        Menentukan aksi trading (0=Hold, 1=Buy, 2=Sell) 
        berdasarkan output NN + rule zona Supply/Demand.
        """
        # Output NN
        nn_output = feed_forward(state, individual)
        predicted_action = np.argmax(nn_output)  # 0=Hold, 1=Buy, 2=Sell
        
        # Cek zona
        in_supply = self.is_in_supply_zone(current_price)
        in_demand = self.is_in_demand_zone(current_price)
        in_liquidity = self.is_in_liquidity_zone(current_price)
        
        # --------------------------------------------
        # LOGIKA KOMBINASI:
        #  - Jika NN bilang BUY, tapi kita di supply => HOLD
        #  - Jika NN bilang SELL, tapi kita di demand => HOLD
        #  - Jika NN bilang HOLD, tapi kita di demand/liquidity => BUY
        #    atau di supply => SELL, sisanya tetap HOLD.
        # --------------------------------------------
        if predicted_action == 1:  # NN -> BUY
            if in_supply:
                action = 0  # hindari buy di supply
            else:
                action = 1
        elif predicted_action == 2:  # NN -> SELL
            if in_demand:
                action = 0  # hindari sell di demand
            else:
                action = 2
        else:
            # predicted_action == 0 => HOLD
            if in_demand or in_liquidity:
                action = 1  # BUY
            elif in_supply:
                action = 2  # SELL
            else:
                action = 0  # HOLD

        return action

    # ================ Fungsi Evaluate: Simulasi Trading & Fitness ================
    def evaluate(self, individual):
        balance = self.initial_money

        # Variabel untuk Drawdown, Winrate, dsb.
        peak_equity = balance
        max_drawdown = 0
        closed_trades = 0
        winning_trades = 0
        positions_durations = []

        # Tracking trade
        inventory = []
        total_trades = 0
        total_profit = 0
        total_loss = 0
        
        state = self.get_state(0)
        cooldown_period = 8
        last_trade_closed = -cooldown_period
        max_trades = 10000
        max_open_positions = 5
        risk_reward_ratio = 5
        compound_factor = 0.05

        BARS_PER_DAY = 96
        MAX_BARS_OPEN = 5 * 96

        for t in range(0, len(self.trend) - 1):
            price = self.trend[t][0]  # close_M15
            action = self.act(individual, state, price)
            next_state = self.get_state(t + 1)

            # Perhitungan lot (persentase balance)
            trade_amount = ((1 + compound_factor) * 0.15) * balance

            # Hitung ATR sederhana (std dev dari 96 bar terakhir)
            recent_bars = self.trend[max(0, t-96):t, 0]
            atr = np.std(recent_bars) if len(recent_bars) > 0 else 1.0

            stop_loss = price - atr * 3.5
            take_profit = price + (atr * 3.5 * risk_reward_ratio)
            trailing_stop = stop_loss

            # Buka posisi (Buy/Sell) jika dibolehkan
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
                    total_trades += 1

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
                    total_trades += 1

            # Hitung floating P/L untuk drawdown
            unrealized_PL = 0
            for pos in inventory:
                open_price = pos['open_price']
                invested_amount = pos['invested_amount']
                position_type = pos['position_type']
                if position_type == "BUY":
                    unrealized_PL += ((price - open_price) / open_price) * invested_amount
                else:  # SELL
                    unrealized_PL += ((open_price - price) / open_price) * invested_amount

            equity = balance + unrealized_PL
            peak_equity = max(peak_equity, equity)
            current_drawdown = (peak_equity - equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, current_drawdown)

            # Cek SL/TP/trailing stop/durasi
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
                    if price_now > open_price:
                        new_trailing_stop = price_now - atr * 2
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_trailing_stop)
                else:  # SELL
                    if price_now < open_price:
                        new_trailing_stop = price_now + atr * 2
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_trailing_stop)

                # Kondisi close
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

        # Tutup posisi tersisa di akhir
        final_price = self.trend[-1][0]
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

            duration_in_bars = final_t - pos['open_t']
            positions_durations.append(duration_in_bars / BARS_PER_DAY)
            total_trades += 1
            print(f"⚠️ Training berakhir, posisi terakhir ditutup di harga {final_price:.2f} dengan profit {profit_trade:.2f}")

        # Hitung profit (dollar dan %)
        profit_dollar = balance - self.initial_money
        profit_percent = (profit_dollar / self.initial_money) * 100

        # -- Ubah Penalti jika closed_trades == 0 --
        if closed_trades == 0:
            # Tidak ada trade tertutup => Penalti lebih ringan daripada -1000%
            profit_percent = -50
            profit_dollar = -(0.5 * self.initial_money)
            winrate = 0
            avg_duration = 0
            max_duration = 0
        else:
            winrate = (winning_trades / closed_trades) * 100
            avg_duration = sum(positions_durations) / len(positions_durations)
            max_duration = max(positions_durations)

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

    # ================ Fungsi Evolve (GA) ================
    def evolve(self, generations=50):
        """
        Proses evolusi GA untuk menemukan model terbaik.
        """
        previous_fitness = None

        for epoch in range(generations):
            print(f"\n🔵 Epoch {epoch+1}/{generations} sedang berjalan...")
            for individual in self.population:
                self.evaluate(individual)

            # Urutkan populasi berdasarkan fitness tertinggi
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            fittest = self.population[0]

            avg_fitness = np.mean([ind.fitness for ind in self.population])
            worst_fitness = self.population[-1].fitness

            print(f"✅ Epoch {epoch+1} - Fitness Terbaik: {fittest.fitness:.2f}, Rata-rata: {avg_fitness:.2f}, Terburuk: {worst_fitness:.2f}")
            logging.info(f"Epoch {epoch+1}, Fitness Terbaik: {fittest.fitness:.2f}, Rata-rata: {avg_fitness:.2f}, Terburuk: {worst_fitness:.2f}")

            # Jika fitness turun drastis, kecilkan mutation_rate
            if epoch > 1 and previous_fitness is not None:
                if fittest.fitness < previous_fitness * 0.5:
                    print("⚠️ Fitness turun drastis, mengurangi mutasi!")
                    logging.info("⚠️ Fitness turun drastis, mengurangi mutasi!")
                    self.mutation_rate *= 0.8

            previous_fitness = fittest.fitness

            # Simpan model terbaik
            model_path = "../models/best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(fittest, f)

            # Buat populasi baru (elitisme + crossover + mutasi)
            new_population = [fittest]  # elitisme

            for _ in range(self.population_size - 1):
                parent1, parent2 = np.random.choice(self.population[:10], 2)
                child = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child))

            self.population = new_population

        return self.population[0]

# ======================================================
# 7. Fungsi Simulasi Trading dengan Model Terbaik
# ======================================================
def simulate_trading(model, trend, window_size, initial_money, num_features, supply_zones, demand_zones):
    balance = initial_money
    inventory = []
    buy_points = []
    sell_points = []
    state = None
    cooldown_period = 8
    last_trade_closed = -cooldown_period
    BARS_PER_DAY = 96
    max_open_positions = 5
    max_trades = 10000
    compound_factor = 0.05
    risk_reward_ratio = 5
    max_bars_open = 5 * BARS_PER_DAY

    # Buat instance NeuroEvolution "dummy" untuk memanfaatkan method "act" 
    # (TANPA mutasi/crossover)
    neuro_evolve_instance = NeuroEvolution(
        population_size=1,
        mutation_rate=0.0,  # no mutation saat simulasi
        model_generator=lambda id_: model, 
        trend=trend,
        initial_money=initial_money,
        num_features=num_features,
        supply_zones=supply_zones,
        demand_zones=demand_zones
    )

    for t in range(len(trend) - 1):
        price = trend[t][0]  # close_M15

        # Ambil window state
        if t < window_size:
            window = np.zeros((window_size, num_features))
            if t > 0:
                window[-t:] = trend[:t]
        else:
            window = trend[t - window_size:t]
        state = window.flatten()

        # Dapatkan aksi dari method 'act' di NeuroEvolution
        action = neuro_evolve_instance.act(model, state, price)

        # Hitung ATR sederhana
        recent_bars = trend[max(0, t-96):t, 0]
        atr = np.std(recent_bars) if len(recent_bars) > 0 else 1.0

        trade_amount = ((1 + compound_factor) * 0.15) * balance
        stop_loss = price - atr * 3.5
        take_profit = price + (atr * 3.5 * risk_reward_ratio)
        trailing_stop = stop_loss

        # Buka posisi jika diperbolehkan
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

        # Cek SL/TP/trailing stop/max durasi
        for pos in inventory[:]:
            open_price = pos['open_price']
            invested_amount = pos['invested_amount']
            position_type = pos['position_type']
            open_t = pos['open_t']

            # Update trailing stop
            if position_type == "BUY":
                if price > open_price:
                    new_trailing_stop = price - atr * 2
                    pos['trailing_stop'] = max(pos['trailing_stop'], new_trailing_stop)
            else:
                if price < open_price:
                    new_trailing_stop = price + atr * 2
                    pos['trailing_stop'] = min(pos['trailing_stop'], new_trailing_stop)

            close_position = False
            duration_in_bars = t - open_t

            if duration_in_bars >= max_bars_open:
                close_position = True
            elif position_type == "BUY":
                if price <= pos['trailing_stop'] or price >= pos['take_profit']:
                    close_position = True
            else:  # SELL
                if price >= pos['trailing_stop'] or price <= pos['take_profit']:
                    close_position = True

            if close_position:
                if position_type == "BUY":
                    profit_trade = ((price - open_price) / open_price) * invested_amount
                else:
                    profit_trade = ((open_price - price) / open_price) * invested_amount

                balance += invested_amount + profit_trade
                inventory.remove(pos)
                last_trade_closed = t

        # Cek batas total trades
        if len(buy_points) + len(sell_points) >= max_trades:
            print(f"⚠️ Mencapai batas {max_trades} transaksi. Menghentikan trading!")
            break

        # Jika saldo habis
        if balance <= 0:
            print("⚠️ Saldo habis. Menghentikan trading lebih awal.")
            break

    # Tutup posisi di akhir
    final_price = trend[-1][0]
    final_t = len(trend) - 1
    while len(inventory) > 0:
        pos = inventory.pop(0)
        open_price = pos['open_price']
        invested_amount = pos['invested_amount']
        position_type = pos['position_type']

        if position_type == "BUY":
            profit_trade = ((final_price - open_price) / open_price) * invested_amount
            sell_points.append(final_t)
        else:
            profit_trade = ((open_price - final_price) / open_price) * invested_amount
            buy_points.append(final_t)

        balance += invested_amount + profit_trade

    return buy_points, sell_points

# ======================================================
# 8. Plot Hasil Trading
# ======================================================
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
    for zone, freq in supply_zones:
        plt.axhline(y=zone, color='magenta', linestyle='--', linewidth=0.8)
        plt.text(df['time'].iloc[len(df)//2], zone, f'Supply ({freq})', 
                 color='magenta', fontsize=8, verticalalignment='bottom')

    # Plot Demand Zones
    for zone, freq in demand_zones:
        plt.axhline(y=zone, color='cyan', linestyle='--', linewidth=0.8)
        plt.text(df['time'].iloc[len(df)//2], zone, f'Demand ({freq})', 
                 color='cyan', fontsize=8, verticalalignment='top')

    plt.title('Hasil Trading (NN + Kombinasi Zona Supply/Demand)')
    plt.xlabel('Waktu')
    plt.ylabel('Harga Close M15')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ======================================================
# 9. Eksekusi Utama: Evolve -> Simulate -> Plot
# ======================================================
if __name__ == "__main__":
    print("Supply Zones:", supply_zones)
    print("Demand Zones:", demand_zones)

    # 1. Inisialisasi GA
    neural_evolve = NeuroEvolution(
        population_size=population_size,
        mutation_rate=mutation_rate,
        model_generator=NeuralNetwork,
        trend=feature_data,
        initial_money=initial_money,
        num_features=num_features,
        supply_zones=supply_zones,
        demand_zones=demand_zones
    )

    # 2. Evolusi untuk mencari model terbaik
    best_model = neural_evolve.evolve(generations)

    print(f"\n🚀 Model terbaik disimpan dengan fitness {best_model.fitness:.2f}")
    logging.info(f"Model terbaik disimpan dengan fitness {best_model.fitness:.2f}")

    # 3. Simulasikan trading dengan model terbaik
    buy_points, sell_points = simulate_trading(best_model, feature_data, window_size, 
                                               initial_money, num_features, supply_zones, demand_zones)

    # 4. Plot hasil trading
    plot_trading(df, buy_points, sell_points, supply_zones, demand_zones)
