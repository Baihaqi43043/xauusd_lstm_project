import numpy as np
import pandas as pd
import ta
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

# **1. Inisialisasi Colorama untuk Pewarnaan Terminal**
init(autoreset=True)

# **2. Definisi Kelas NeuralNetwork**
class NeuralNetwork:
    def __init__(self, id_, input_size, hidden_size=128, output_size=3):
        self.id = id_
        # Inisialisasi bobot dengan distribusi normal yang beragam
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        # Atribut tambahan untuk menyimpan metrik
        self.fitness = 0.0
        self.winrate = 0.0
        self.avg_duration = 0.0
        self.max_duration = 0.0
        self.total_loss = 0.0


# **3. Definisi Fungsi Aktivasi**
def relu(X):
    return np.maximum(X, 0)

def softmax(X):
    e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def feed_forward(X, net):
    a1 = np.dot(X, net.W1)
    z1 = relu(a1)
    a2 = np.dot(z1, net.W2)
    return softmax(a2)

# **4. Definisi Kelas NeuroEvolution**
class NeuroEvolution:
    def __init__(self, population_size, mutation_rate, model_generator, initial_money, num_features, window_size, max_trades=700):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_generator = model_generator
        self.initial_money = initial_money
        self.num_features = num_features
        self.window_size = window_size
        self.max_trades = max_trades
        # Inisialisasi populasi dengan model yang dihasilkan oleh model_generator
        self.population = [self.model_generator(i, window_size * num_features) for i in range(population_size)]
    
    def crossover(self, parent1, parent2):
        child = self.model_generator(parent1.id + 1000, self.window_size * self.num_features)
        # Crossover untuk W1
        mask_W1 = np.random.rand(*child.W1.shape) > 0.5
        child.W1 = np.where(mask_W1, parent1.W1, parent2.W1)
        # Crossover untuk W2
        mask_W2 = np.random.rand(*child.W2.shape) > 0.5
        child.W2 = np.where(mask_W2, parent1.W2, parent2.W2)
        return child

    
    def mutate(self, individual):
        # Mutasi untuk W1
        mutation_mask_W1 = np.random.binomial(1, self.mutation_rate, size=individual.W1.shape)
        individual.W1 += np.random.normal(loc=0, scale=0.1, size=individual.W1.shape) * mutation_mask_W1
        # Mutasi untuk W2
        mutation_mask_W2 = np.random.binomial(1, self.mutation_rate, size=individual.W2.shape)
        individual.W2 += np.random.normal(loc=0, scale=0.1, size=individual.W2.shape) * mutation_mask_W2
        return individual

    
    def evaluate(self, individual, data):
        capital = self.initial_money
        position = 0  # 1 = Long, -1 = Short, 0 = No Position
        entry_price = 0
        max_drawdown = 0
        peak = capital
        total_profit_dollar = 0  # Total Profit dalam Dolar
        trades = 0
        total_loss_dollar = 0.0  # Total Loss dalam Dolar
        durations = []
        wins = 0
        capital_history = []  # Menambahkan kapital history

        for i in range(self.window_size, len(data)):
            if trades >= self.max_trades:
                break  # Hentikan evaluasi jika sudah mencapai max_trades

            window_data = data[i - self.window_size:i]
            X = window_data.flatten().reshape(1, -1)
            action_prob = feed_forward(X, individual)[0]
            action = np.argmax(action_prob) - 1  # 0: Sell, 1: Hold, 2: Buy -> mapped to -1, 0, 1

            current_price = data[i, 0]  # Close price

            # Implement simple trading logic
            if action == 1 and position == 0:
                # Buy
                position = 1
                entry_price = current_price
                trades += 1
                trade_start_time = i  # Ganti dengan waktu jika tersedia
            elif action == -1 and position == 0:
                # Sell
                position = -1
                entry_price = current_price
                trades += 1
                trade_start_time = i  # Ganti dengan waktu jika tersedia
            elif position == 1:
                # Implement take-profit and stop-loss for Long with 1:2 ratio
                take_profit = entry_price * 1.02  # 2% profit
                stop_loss = entry_price * 0.99    # 1% loss
                if current_price >= take_profit or current_price <= stop_loss:
                    profit_percentage = (current_price - entry_price) / entry_price * 100
                    if current_price >= take_profit:
                        profit_dollar = self.initial_money * 0.01 * (2 / 100)  # TP 2%
                        total_profit_dollar += profit_dollar
                        wins += 1
                    else:
                        loss_dollar = self.initial_money * 0.01 * (1 / 100)   # SL 1%
                        total_loss_dollar += loss_dollar
                    capital += profit_dollar if current_price >= take_profit else -loss_dollar
                    position = 0
                    trade_end_time = i  # Ganti dengan waktu jika tersedia
                    duration = (trade_end_time - trade_start_time) / (60*60*24)  # dalam hari
                    durations.append(duration)
            elif position == -1:
                # Implement take-profit and stop-loss for Short with 1:2 ratio
                take_profit = entry_price * 0.99  # 1% profit
                stop_loss = entry_price * 1.02    # 2% loss
                if current_price <= take_profit or current_price >= stop_loss:
                    profit_percentage = (entry_price - current_price) / entry_price * 100
                    if current_price <= take_profit:
                        profit_dollar = self.initial_money * 0.01 * (1 / 100)  # TP 1%
                        total_profit_dollar += profit_dollar
                        wins += 1
                    else:
                        loss_dollar = self.initial_money * 0.01 * (2 / 100)   # SL 2%
                        total_loss_dollar += loss_dollar
                    capital += profit_dollar if current_price <= take_profit else -loss_dollar
                    position = 0
                    trade_end_time = i  # Ganti dengan waktu jika tersedia
                    duration = (trade_end_time - trade_start_time) / (60*60*24)  # dalam hari

            # Update peak dan drawdown
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            # Tambahkan kapital ke history
            capital_history.append(capital)

# Hitung metrik tambahan
        if len(capital_history) > 1:
            returns = np.diff(capital_history) / capital_history[:-1]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return != 0 else 0
            # Tambahkan profit factor
            profit_factor = total_profit_dollar / total_loss_dollar if total_loss_dollar != 0 else 0
            # Tambahkan Sortino Ratio (menggunakan downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
            sortino_ratio = mean_return / downside_std if downside_std != 0 else 0
            fitness = (sharpe_ratio * 100) + (sortino_ratio * 50) + (profit_factor * 10) - (max_drawdown * 2)
        else:
            fitness = 0

        # Hitung net profit dan profit percentage
        net_profit = total_profit_dollar - total_loss_dollar
        profit_percentage = (net_profit / self.initial_money) * 100

        # Hitung winrate
        winrate = (wins / trades * 100) if trades > 0 else 0.0

        # Update atribut individu
        individual.fitness = fitness
        individual.winrate = winrate
        individual.avg_duration = np.mean(durations) if durations else 0.0
        individual.max_duration = np.max(durations) if durations else 0.0
        individual.total_loss = total_loss_dollar  # Sesuaikan sesuai perhitungan loss

        # Pewarnaan output untuk profit positif/negatif
        if net_profit > 0:
            profit_color = Fore.GREEN
        else:
            profit_color = Fore.RED

        # Ringkasan hasil individu
        print(
            f"Individu {individual.id} - "
            f"Profit: {profit_color}${net_profit:.2f} ({profit_percentage:.2f}%), "
            f"Winrate: {winrate:.2f}%, "
            f"Closed Trades: {trades}, "
            f"Max Floating DD: {max_drawdown:.2f}%, "
            f"Avg Duration (days): {individual.avg_duration:.2f}, "
            f"Max Duration (days): {individual.max_duration:.2f}, "
            f"Total Profit: ${total_profit_dollar:.2f}, "
            f"Total Loss: ${total_loss_dollar:.2f}{Style.RESET_ALL}"
        )

    def evolve(self, generations, train_data, early_stopping=10):
        best_fitness_history = []
        no_improvement = 0
        best_fitness = -np.inf
        best_model = None
        
        for epoch in range(generations):
            # Evaluasi setiap individu
            for individual in self.population:
                fitness = self.evaluate(individual, train_data)
        
            # Sort populasi berdasarkan fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            current_best = self.population[0]
            best_fitness_history.append(current_best.fitness)
        
            print(f"Generasi {epoch+1}/{generations} - Fitness Terbaik: {current_best.fitness:.2f}%")
        
            # Early Stopping
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                no_improvement = 0
                best_model = current_best
            else:
                no_improvement += 1
                if no_improvement >= early_stopping:
                    print("Early stopping: Tidak ada peningkatan fitness selama beberapa generasi.")
                    break
        
            # Seleksi elit (misalnya, 20% terbaik)
            elite_size = max(1, int(0.2 * self.population_size))
            elites = self.population[:elite_size]
        
            # Ciptakan populasi baru
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = np.random.choice(elites, 2, replace=False)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
        
            # Tambahkan beberapa individu secara acak untuk menjaga diversitas
            if len(new_population) < self.population_size:
                additional = self.population[np.random.choice(len(self.population), self.population_size - len(new_population), replace=False)]
                new_population.extend(additional)
        
            self.population = new_population
        
        print(f"Evolusi selesai. Model terbaik memiliki Fitness: {best_model.fitness:.2f}%")
        return best_model, best_fitness_history


# **5. Membaca dan Preprocessing Data**
def add_indicators(df):
    required_columns = ['close_M15', 'high_M15', 'low_M15']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam DataFrame.")
    
    df['EMA_200'] = ta.trend.EMAIndicator(df['close_M15'], window=200).ema_indicator()
    df['RSI_14'] = ta.momentum.RSIIndicator(df['close_M15'], window=14).rsi()
    df['ATR_14'] = ta.volatility.AverageTrueRange(df['high_M15'], df['low_M15'], df['close_M15'], window=14).average_true_range()
    df['ADX_14'] = ta.trend.ADXIndicator(df['high_M15'], df['low_M15'], df['close_M15'], window=14).adx()
    bb = ta.volatility.BollingerBands(df['close_M15'], window=20)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # Menambahkan MACD
    macd = ta.trend.MACD(df['close_M15'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Menambahkan Stochastic Oscillator
    stochastic = ta.momentum.StochasticOscillator(df['high_M15'], df['low_M15'], df['close_M15'], window=14)
    df['Stochastic'] = stochastic.stoch()
    df['Stochastic_Signal'] = stochastic.stoch_signal()
    
    # Cek keberadaan kolom 'volume' sebelum menambahkannya
    if 'volume' in df.columns:
        df['Volume_Avg'] = ta.volume.VolumeWeightedAveragePrice(df['high_M15'], df['low_M15'], df['close_M15'], df['volume'], window=14).volume_weighted_average_price()
    else:
        print("Warning: Kolom 'volume' tidak ditemukan. Indikator 'Volume_Avg' tidak ditambahkan.")
    
    # Mengisi nilai NaN jika ada
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    return df

from sklearn.preprocessing import StandardScaler

def prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    print(f"Kolom dalam DataFrame: {df.columns.tolist()}")
    print("Contoh data:")
    print(df.head())
    
    # Memperluas periode data (misalnya, dari 2020-01-01 hingga 2024-12-31)
    df = df[(df['time'] >= '2020-01-01') & (df['time'] <= '2024-12-31')]
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = add_indicators(df)
    
    # Memilih fitur yang akan digunakan
    features = [
        'close_M15', 'RSI_14', 'ATR_14', 'EMA_200', 'ADX_14',
        'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Diff',
        'Stochastic', 'Stochastic_Signal'
    ]
    
    # Jika 'Volume_Avg' ditambahkan, tambahkan juga ke fitur
    if 'Volume_Avg' in df.columns:
        features.append('Volume_Avg')
    
    num_features = len(features)  # Menyesuaikan jumlah fitur
    
    # Membagi data menjadi training dan testing
    train_df = df[df['time'] < '2024-01-01'].copy()
    test_df = df[df['time'] >= '2024-01-01'].copy()
    
    # Mengubah data menjadi numpy array untuk efisiensi
    train_data = train_df[features].values
    test_data = test_df[features].values
    
    # Normalisasi data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    
    return train_data, test_data, features, num_features

# **6. Membuat Model Generator**
def model_generator(id_, input_size):
    return NeuralNetwork(id_, input_size)

# **7. Backtest Function**
def backtest_model(model, data, initial_money, window_size, features):
    capital = initial_money
    position = 0
    entry_price = 0
    capital_history = []
    drawdown_history = []
    peak = capital
    max_drawdown = 0
    total_profit_dollar = 0
    trades = 0
    total_loss_dollar = 0.0
    durations = []
    wins = 0

    risk_per_trade = 0.01  # 1% risiko per perdagangan

    for i in range(window_size, len(data)):
        if trades >= 700:
            break  # Hentikan backtest jika sudah mencapai 700 perdagangan

        window_data = data[i - window_size:i]
        X = window_data.flatten().reshape(1, -1)
        action_prob = feed_forward(X, model)[0]
        action = np.argmax(action_prob) - 1  # -1: Sell, 0: Hold, 1: Buy

        current_price = data[i, 0]  # Close price

        # Implement simple trading logic
        if action == 1 and position == 0:
            # Buy
            position = 1
            entry_price = current_price
            trades += 1
            trade_start_time = i  # Ganti dengan waktu jika tersedia
        elif action == -1 and position == 0:
            # Sell
            position = -1
            entry_price = current_price
            trades += 1
            trade_start_time = i  # Ganti dengan waktu jika tersedia
        elif position == 1:
            # Implement take-profit and stop-loss for Long with 1:2 ratio
            take_profit = entry_price * 1.02  # 2% profit
            stop_loss = entry_price * 0.99    # 1% loss
            if current_price >= take_profit or current_price <= stop_loss:
                profit_percentage = (current_price - entry_price) / entry_price * 100
                if current_price >= take_profit:
                    profit_dollar = initial_money * risk_per_trade * (2 / 100)  # TP 2%
                    total_profit_dollar += profit_dollar
                    wins += 1
                else:
                    loss_dollar = initial_money * risk_per_trade * (1 / 100)   # SL 1%
                    total_loss_dollar += loss_dollar
                capital += profit_dollar if current_price >= take_profit else -loss_dollar
                position = 0
                trade_end_time = i  # Ganti dengan waktu jika tersedia
                duration = (trade_end_time - trade_start_time) / (60*60*24)  # dalam hari
                durations.append(duration)
        elif position == -1:
            # Implement take-profit and stop-loss for Short with 1:2 ratio
            take_profit = entry_price * 0.99  # 1% profit
            stop_loss = entry_price * 1.02    # 2% loss
            if current_price <= take_profit or current_price >= stop_loss:
                profit_percentage = (entry_price - current_price) / entry_price * 100
                if current_price <= take_profit:
                    profit_dollar = initial_money * risk_per_trade * (1 / 100)  # TP 1%
                    total_profit_dollar += profit_dollar
                    wins += 1
                else:
                    loss_dollar = initial_money * risk_per_trade * (2 / 100)   # SL 2%
                    total_loss_dollar += loss_dollar
                capital += profit_dollar if current_price <= take_profit else -loss_dollar
                position = 0
                trade_end_time = i  # Ganti dengan waktu jika tersedia
                duration = (trade_end_time - trade_start_time) / (60*60*24)  # dalam hari

        # Update peak dan drawdown
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # Tambahkan kapital ke history
        capital_history.append(capital)
        drawdown_history.append(drawdown)

    # Hitung metrik tambahan
    if len(capital_history) > 1:
        returns = np.diff(capital_history) / capital_history[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return if std_return != 0 else 0
        # Tambahkan profit factor
        profit_factor = total_profit_dollar / total_loss_dollar if total_loss_dollar != 0 else 0
        fitness = sharpe_ratio * 100 + profit_factor * 10 - max_drawdown
    else:
        fitness = 0

    # Hitung net profit dan profit percentage
    net_profit = total_profit_dollar - total_loss_dollar
    profit_percentage = (net_profit / initial_money) * 100

    # Hitung winrate
    winrate = (wins / trades * 100) if trades > 0 else 0.0

    # Ringkasan hasil backtest
    print("\n=== Hasil Backtest ===")
    if net_profit > 0:
        profit_color = Fore.GREEN
    else:
        profit_color = Fore.RED

    print(
        f"Net Profit: {profit_color}${net_profit:.2f} ({profit_percentage:.2f}%){Style.RESET_ALL}"
    )
    print(f"Winrate: {winrate:.2f}%")
    print(f"Closed Trades: {trades}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Avg Duration (days): {np.mean(durations) if durations else 0.0:.2f}")
    print(f"Max Duration (days): {np.max(durations) if durations else 0.0:.2f}")
    print(f"Total Profit: ${total_profit_dollar:.2f}")
    print(f"Total Loss: ${total_loss_dollar:.2f}")

    return capital_history, drawdown_history, total_profit_dollar, max_drawdown, trades

# **8. Membaca dan Preprocessing Data**
def prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    print(f"Kolom dalam DataFrame: {df.columns.tolist()}")
    print("Contoh data:")
    print(df.head())
    
    # Memperluas periode data (misalnya, dari 2020-01-01 hingga 2024-12-31)
    df = df[(df['time'] >= '2020-01-01') & (df['time'] <= '2024-12-31')]
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = add_indicators(df)
    
    # Memilih fitur yang akan digunakan
    features = [
        'close_M15', 'RSI_14', 'ATR_14', 'EMA_200', 'ADX_14',
        'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal', 'MACD_Diff',
        'Stochastic', 'Stochastic_Signal'
    ]
    
    # Jika 'Volume_Avg' ditambahkan, tambahkan juga ke fitur
    if 'Volume_Avg' in df.columns:
        features.append('Volume_Avg')
    
    num_features = len(features)  # Menyesuaikan jumlah fitur
    
    # Membagi data menjadi training dan testing
    train_df = df[df['time'] < '2024-01-01'].copy()
    test_df = df[df['time'] >= '2024-01-01'].copy()
    
    # Mengubah data menjadi numpy array untuk efisiensi
    train_data = train_df[features].values
    test_data = test_df[features].values
    
    return train_data, test_data, features, num_features

# **9. Membuat Model Generator**
def model_generator(id_, input_size):
    return NeuralNetwork(id_, input_size)

# **10. Backtest Function**
def backtest_model(model, data, initial_money, window_size, features):
    capital = initial_money
    position = 0
    entry_price = 0
    capital_history = []
    drawdown_history = []
    peak = capital
    max_drawdown = 0
    total_profit_dollar = 0
    trades = 0
    total_loss_dollar = 0.0
    durations = []
    wins = 0

    risk_per_trade = 0.01  # 1% risiko per perdagangan

    for i in range(window_size, len(data)):
        if trades >= 700:
            break  # Hentikan backtest jika sudah mencapai 700 perdagangan

        window_data = data[i - window_size:i]
        X = window_data.flatten().reshape(1, -1)
        action_prob = feed_forward(X, model)[0]
        action = np.argmax(action_prob) - 1  # -1: Sell, 0: Hold, 1: Buy

        current_price = data[i, 0]  # Close price

        # Implement simple trading logic
        if action == 1 and position == 0:
            # Buy
            position = 1
            entry_price = current_price
            trades += 1
            trade_start_time = i  # Ganti dengan waktu jika tersedia
        elif action == -1 and position == 0:
            # Sell
            position = -1
            entry_price = current_price
            trades += 1
            trade_start_time = i  # Ganti dengan waktu jika tersedia
        elif position == 1:
            # Implement take-profit and stop-loss for Long with 1:2 ratio
            take_profit = entry_price * 1.02  # 2% profit
            stop_loss = entry_price * 0.99    # 1% loss
            if current_price >= take_profit or current_price <= stop_loss:
                profit_percentage = (current_price - entry_price) / entry_price * 100
                if current_price >= take_profit:
                    profit_dollar = initial_money * risk_per_trade * (2 / 100)  # TP 2%
                    total_profit_dollar += profit_dollar
                    wins += 1
                else:
                    loss_dollar = initial_money * risk_per_trade * (1 / 100)   # SL 1%
                    total_loss_dollar += loss_dollar
                capital += profit_dollar if current_price >= take_profit else -loss_dollar
                position = 0
                trade_end_time = i  # Ganti dengan waktu jika tersedia
                duration = (trade_end_time - trade_start_time) / (60*60*24)  # dalam hari
                durations.append(duration)
        elif position == -1:
            # Implement take-profit and stop-loss for Short with 1:2 ratio
            take_profit = entry_price * 0.99  # 1% profit
            stop_loss = entry_price * 1.02    # 2% loss
            if current_price <= take_profit or current_price >= stop_loss:
                profit_percentage = (entry_price - current_price) / entry_price * 100
                if current_price <= take_profit:
                    profit_dollar = initial_money * risk_per_trade * (1 / 100)  # TP 1%
                    total_profit_dollar += profit_dollar
                    wins += 1
                else:
                    loss_dollar = initial_money * risk_per_trade * (2 / 100)   # SL 2%
                    total_loss_dollar += loss_dollar
                capital += profit_dollar if current_price <= take_profit else -loss_dollar
                position = 0
                trade_end_time = i  # Ganti dengan waktu jika tersedia
                duration = (trade_end_time - trade_start_time) / (60*60*24)  # dalam hari

        # Update peak dan drawdown
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # Tambahkan kapital ke history
        capital_history.append(capital)
        drawdown_history.append(drawdown)

    # Hitung metrik tambahan
    if len(capital_history) > 1:
        returns = np.diff(capital_history) / capital_history[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return if std_return != 0 else 0
        # Tambahkan profit factor
        profit_factor = total_profit_dollar / total_loss_dollar if total_loss_dollar != 0 else 0
        fitness = sharpe_ratio * 100 + profit_factor * 10 - max_drawdown
    else:
        fitness = 0

    # Hitung net profit dan profit percentage
    net_profit = total_profit_dollar - total_loss_dollar
    profit_percentage = (net_profit / initial_money) * 100

    # Hitung winrate
    winrate = (wins / trades * 100) if trades > 0 else 0.0

    # Ringkasan hasil backtest
    print("\n=== Hasil Backtest ===")
    if net_profit > 0:
        profit_color = Fore.GREEN
    else:
        profit_color = Fore.RED

    print(
        f"Net Profit: {profit_color}${net_profit:.2f} ({profit_percentage:.2f}%)"
        f"{Style.RESET_ALL}"
    )
    print(f"Winrate: {winrate:.2f}%")
    print(f"Closed Trades: {trades}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Avg Duration (days): {np.mean(durations) if durations else 0.0:.2f}")
    print(f"Max Duration (days): {np.max(durations) if durations else 0.0:.2f}")
    print(f"Total Profit: ${total_profit_dollar:.2f}")
    print(f"Total Loss: ${total_loss_dollar:.2f}")

    return capital_history, drawdown_history, total_profit_dollar, max_drawdown, trades

# **11. Main Function**
def main():
    # **a. Membaca dan Menyiapkan Data**
    data_file = '../data/gold_usd_preprocessed.csv'  # Sesuaikan path sesuai kebutuhan
    train_data, test_data, features, num_features = prepare_data(data_file)
    
    # **b. Parameter Trading**
    initial_money = 50000  # Modal awal
    window_size = 96        # 24 jam data (96 * 15 menit)
    
    # **c. Inisialisasi NeuroEvolution**
    neural_evolve = NeuroEvolution(
        population_size=50,          # Ukuran populasi
        mutation_rate=0.05,          # Tingkat mutasi
        model_generator=model_generator,
        initial_money=initial_money,
        num_features=num_features,
        window_size=window_size
    )
    
    # **d. Jalankan Evolusi**
    generations = 50
    best_model, best_fitness_history = neural_evolve.evolve(generations=generations, train_data=train_data)
    
    print(f"\nEvolusi selesai. Model terbaik memiliki Fitness: {best_model.fitness:.2f}%")
    
    # **e. Backtest pada Data Testing**
    capital_history, drawdown_history, total_profit_dollar, max_drawdown, trades = backtest_model(
        best_model,
        test_data,
        initial_money,
        window_size,
        features
    )
    
    # **f. Visualisasi Equity Curve dan Drawdown**
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(capital_history, label='Equity Curve')
    plt.title('Equity Curve')
    plt.xlabel('Trades')
    plt.ylabel('Capital')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(drawdown_history, label='Drawdown', color='red')
    plt.title('Drawdown')
    plt.xlabel('Trades')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # **g. Menyimpan Model Terbaik**
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("\nModel terbaik disimpan sebagai 'best_model.pkl'.")

# **12. Menjalankan Main Function**
if __name__ == "__main__":
    main()
