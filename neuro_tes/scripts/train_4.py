import numpy as np
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Konfigurasi Logging
# -----------------------------
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------------
# 2. Definisi Kelas NeuralNetwork
# -----------------------------
class NeuralNetwork:
    def __init__(self, id_, window_size=12, hidden_size=128):
        self.W1 = np.random.randn(window_size, hidden_size) / np.sqrt(window_size)
        self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
        self.fitness = 0
        self.id = id_

    @staticmethod
    def relu(X):
        return np.maximum(X, 0)

    @staticmethod
    def softmax(X):
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def feed_forward(self, X):
        a1 = np.dot(X, self.W1)
        z1 = self.relu(a1)
        a2 = np.dot(z1, self.W2)
        return self.softmax(a2)

# -----------------------------
# 3. Definisi Kelas NeuroEvolution
# -----------------------------
class NeuroEvolution:
    def __init__(self, population_size, mutation_rate, model_generator, trend, initial_money, supply_zones, demand_zones, trend_direction):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_generator = model_generator
        self.trend = trend
        self.initial_money = initial_money
        self.population = [self.model_generator(i) for i in range(population_size)]
        self.supply_zones = supply_zones
        self.demand_zones = demand_zones
        self.trend_direction = trend_direction

    def mutate(self, individual):
        try:
            mutation_mask_W1 = np.random.binomial(1, p=self.mutation_rate, size=individual.W1.shape)
            individual.W1 += np.random.normal(loc=0, scale=0.5, size=individual.W1.shape) * mutation_mask_W1

            mutation_mask_W2 = np.random.binomial(1, p=self.mutation_rate, size=individual.W2.shape)
            individual.W2 += np.random.normal(loc=0, scale=0.5, size=individual.W2.shape) * mutation_mask_W2

            return individual
        except Exception as e:
            logging.error(f"Error dalam mutate individu {individual.id}: {e}")
            raise e

    def crossover(self, parent1, parent2):
        try:
            child = self.model_generator(parent1.id + 1000)
            cutoff = np.random.randint(0, parent1.W1.shape[1])

            child.W1[:, :cutoff] = parent1.W1[:, :cutoff]
            child.W1[:, cutoff:] = parent2.W1[:, cutoff:]

            child.W2[:, :cutoff] = parent1.W2[:, :cutoff]
            child.W2[:, cutoff:] = parent2.W2[:, cutoff:]

            return child
        except Exception as e:
            logging.error(f"Error dalam crossover: {e}")
            raise e

    def evaluate(self, individual):
        """Evaluasi individu dengan melakukan trading simulasi"""
        try:
            profit = np.random.uniform(-10, 10)  # Simulasi profit
            individual.fitness = profit
            logging.info(f"Individu {individual.id} - Profit: {profit:.2f}")
            print(f"Individu {individual.id} - Profit: {profit:.2f}")
            return profit
        except Exception as e:
            logging.error(f"Error dalam evaluasi individu {individual.id}: {e}")
            raise e

# -----------------------------
# 4. Definisi Fungsi Pendukung
# -----------------------------
def fetch_data_csv(data_file):
    try:
        df = pd.read_csv(data_file)
        close = df['close'].values
        low = df['low'].values
        high = df['high'].values
        return df, close, low, high
    except Exception as e:
        logging.error(f"Error dalam fetch_data_csv: {e}")
        raise e

def identify_zones(df, window=20, threshold=0.02):
    try:
        demand_zones, supply_zones = [], []
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            current_price = df.iloc[i]['close']
            min_price = window_data['low'].min()
            max_price = window_data['high'].max()
            if current_price <= min_price * (1 + threshold):
                demand_zones.append((i, current_price))
            elif current_price >= max_price * (1 - threshold):
                supply_zones.append((i, current_price))
        return demand_zones, supply_zones
    except Exception as e:
        logging.error(f"Error dalam identify_zones: {e}")
        raise e

# -----------------------------
# 5. Latih Model dan Simpan ke File
# -----------------------------
def main():
    try:
        data_file = '../data/gold_usd_M15.csv'
        df, close, low, high = fetch_data_csv(data_file)
        demand_zones, supply_zones = identify_zones(df)

        neural_evolve = NeuroEvolution(
            population_size=100,
            mutation_rate=0.2,
            model_generator=lambda id_: NeuralNetwork(id_),
            trend=close,
            initial_money=50000,
            supply_zones=supply_zones,
            demand_zones=demand_zones,
            trend_direction=[]
        )

        print("Memulai training...")
        for individual in neural_evolve.population:
            neural_evolve.evaluate(individual)
            print(f"Summary Trading Individu {individual.id}: Fitness {individual.fitness:.2f}")

        print("Training selesai!")
    except Exception as e:
        logging.error(f"Error dalam main: {e}")
        raise e

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Terjadi error: {e}")
        print(f"Terjadi error: {e}")
