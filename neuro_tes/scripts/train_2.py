# Import Library
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
import random

# Konfigurasi Logging
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed untuk reproducibility
np.random.seed(42)

# -----------------------------
# 1. Membaca dan Memproses Data
# -----------------------------

data_file = '../data/gold_usd_M15.csv'

# Membaca data CSV
df = pd.read_csv(data_file, parse_dates=['time'])

# Filter hanya untuk tahun 2024
df = df[(df['time'] >= '2024-01-01') & (df['time'] <= '2024-12-31')]

# Pastikan data diurutkan berdasarkan waktu
df.sort_values('time', inplace=True)
df.reset_index(drop=True, inplace=True)

# Mengisi nilai yang hilang
df.ffill(inplace=True)

# Ambil harga penutupan (close) sebagai list
close = df['close'].values.tolist()

print(f"📊 Data setelah pemotongan: {len(df)} baris")
logging.info(f"📊 Data setelah pemotongan: {len(df)} baris")

# -----------------------------
# 2. Definisi Kelas dan Fungsi
# -----------------------------

window_size = 60
initial_money = 50000
population_size = 100
generations = 50
mutation_rate = 0.15

class NeuralNetwork:
    def __init__(self, id_, hidden_size=128):
        self.W1 = np.random.randn(window_size, hidden_size) / np.sqrt(window_size)
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
    def __init__(self, population_size, mutation_rate, model_generator, trend, initial_money):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_generator = model_generator
        self.trend = trend
        self.initial_money = initial_money
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
        return child
    
    def get_state(self, t):
        """Mengambil jendela data harga terakhir untuk diproses oleh model"""
        if t < window_size:
            return np.array([self.trend[:window_size]])
        else:
            return np.array([self.trend[t - window_size:t]])

    def act(self, individual, state):
        """Menentukan aksi trading berdasarkan model"""
        action = np.argmax(feed_forward(state, individual))

        # Paksa model untuk tidak hanya HOLD
        if np.random.rand() < 0.1:
            action = np.random.choice([1, 2])  # Paksa sesekali BUY atau SELL

        return action
    
    import random

    def evaluate(self, individual):
        """Evaluasi fitness berdasarkan total profit dari trading"""
        balance = self.initial_money
        inventory = []  # Hanya bisa ada 1 transaksi dalam satu waktu
        total_trades = 0  
        total_profit = 0  # Akumulasi profit dari semua transaksi
        total_loss = 0  # Akumulasi loss dari semua transaksi
        state = self.get_state(0)
        position_type = None  # Menyimpan apakah posisi yang dibuka adalah BUY atau SELL
        last_trade_closed = 0  # Menyimpan waktu kapan trade terakhir ditutup

        max_trades = 10000  # BATASI JUMLAH TRANSAKSI TOTAL (default: 100)
        risk_reward_ratio = 5  # RR = 1:3 (SL = 2%, TP = 6%)
        cooldown_period = 12  # JEDA ANTARA TRADE BARU (misal: 10 candle)
        
        for t in range(0, len(self.trend) - 1):
            action = self.act(individual, state)
            next_state = self.get_state(t + 1)
            price = self.trend[t]

            # Menentukan lot size (5-15% dari saldo saat ini)
            trade_amount = 0.25 * balance  

            # Gunakan ATR untuk menentukan Stop Loss & Take Profit
            atr = np.std(self.trend[max(0, t-60):t])  # ATR dari 30 candle terakhir
            stop_loss = price - atr * 3.2  # SL = 2 x ATR
            take_profit = price + (atr * 3.2 * risk_reward_ratio)  # TP = 6 x ATR

            # ✅ **Hanya bisa membuka posisi jika sudah melewati cooldown period**
            if len(inventory) == 0 and (t - last_trade_closed) >= cooldown_period:
                if action == 1 and balance >= trade_amount:  # BUY
                    inventory.append((price, trade_amount, "BUY", stop_loss, take_profit))
                    balance -= trade_amount
                    total_trades += 1
                elif action == 2 and balance >= trade_amount:  # SELL
                    inventory.append((price, trade_amount, "SELL", stop_loss, take_profit))
                    balance -= trade_amount
                    total_trades += 1

            # Mengecek apakah posisi harus ditutup berdasarkan SL atau TP
            elif len(inventory) > 0:
                open_price, invested_amount, position_type, sl, tp = inventory[0]
                price_now = self.trend[t]

                # Jika posisi BUY, tutup jika harga naik ke TP atau turun ke SL
                if position_type == "BUY" and (price_now >= tp or price_now <= sl):
                    profit_trade = ((price_now - open_price) / open_price) * invested_amount
                    inventory.pop(0)  # Tutup posisi
                    balance += invested_amount + profit_trade
                    total_profit += max(0, profit_trade)
                    total_loss += abs(min(0, profit_trade))
                    total_trades += 1
                    last_trade_closed = t  # 🚀 Simpan kapan trade terakhir ditutup

                # Jika posisi SELL, tutup jika harga turun ke TP atau naik ke SL
                elif position_type == "SELL" and (price_now <= tp or price_now >= sl):
                    profit_trade = ((open_price - price_now) / open_price) * invested_amount
                    inventory.pop(0)  # Tutup posisi
                    balance += invested_amount + profit_trade
                    total_profit += max(0, profit_trade)
                    total_loss += abs(min(0, profit_trade))
                    total_trades += 1
                    last_trade_closed = t  # 🚀 Simpan kapan trade terakhir ditutup

            # BATASI JUMLAH TRANSAKSI
            if total_trades >= max_trades:
                print(f"⚠️ Individu {individual.id} mencapai batas {max_trades} transaksi. Menghentikan trading!")
                break

            # Jika saldo habis, hentikan trading
            if balance <= 0:
                print(f"⚠️ Individu {individual.id} kehabisan saldo. Menghentikan trading lebih awal.")
                break

            state = next_state

        # 🚨 **PAKSA MENUTUP TRANSAKSI YANG MASIH TERBUKA DI AKHIR TRAINING**
        if len(inventory) > 0:
            open_price, invested_amount, position_type, _, _ = inventory.pop(0)
            final_price = self.trend[-1]  # Gunakan harga terakhir di dataset

            if position_type == "BUY":
                profit_trade = ((final_price - open_price) / open_price) * invested_amount
            elif position_type == "SELL":
                profit_trade = ((open_price - final_price) / open_price) * invested_amount

            balance += invested_amount + profit_trade
            total_profit += max(0, profit_trade)
            total_loss += abs(min(0, profit_trade))
            total_trades += 1

            print(f"⚠️ Training berakhir, posisi terakhir ditutup di harga {final_price:.2f} dengan profit {profit_trade:.2f}")

        # Hitung profit dalam DOLLAR dan PERSENTASE
        profit_dollar = balance - self.initial_money
        profit_percent = (profit_dollar / self.initial_money) * 100  

        # Jika tidak ada transaksi, berikan fitness negatif
        if total_trades == 0:
            profit_percent = -1000
            profit_dollar = -self.initial_money  # Anggap modal hilang semua

        # Logging informasi transaksi
        logging.info(f"Individu {individual.id} - Profit: ${profit_dollar:.2f} ({profit_percent:.2f}%), Total Trades: {total_trades}, Total Profit: ${total_profit:.2f}, Total Loss: ${total_loss:.2f}")
        
        print(f"📊 Individu {individual.id} Summary: Profit ${profit_dollar:.2f} ({profit_percent:.2f}%), Total Trades: {total_trades}, Total Profit: ${total_profit:.2f}, Total Loss: ${total_loss:.2f}")

        individual.fitness = profit_percent  # Gunakan persen untuk fitness
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

            print(f"✅ Epoch {epoch} - Fitness Terbaik: {fittest.fitness:.2f}, Rata-rata: {avg_fitness:.2f}, Terburuk: {worst_fitness:.2f}")
            logging.info(f"Epoch {epoch}, Fitness Terbaik: {fittest.fitness:.2f}, Rata-rata: {avg_fitness:.2f}, Terburuk: {worst_fitness:.2f}")

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
# 4. Latih Model dan Simpan ke File
# -----------------------------

neural_evolve = NeuroEvolution(population_size, mutation_rate, NeuralNetwork, close, initial_money)
best_model = neural_evolve.evolve(generations)

print(f"\n🚀 Model terbaik disimpan dengan fitness {best_model.fitness:.2f}")
logging.info(f"Model terbaik disimpan dengan fitness {best_model.fitness:.2f}")
