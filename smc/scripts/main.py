import random
import numpy as np
from deap import base, creator, tools, algorithms

def genetic_algorithm(X, y, population_size=50, generations=30, crossover_prob=0.5, mutation_prob=0.2):
    """
    Fungsi untuk melakukan optimasi parameter RSI_buy dan RSI_sell
    menggunakan algoritma genetika berbasis DEAP.

    Parameter:
    - X: array fitur (misalnya data indikator teknikal)
    - y: array target (label biner 0 atau 1)
    - population_size: ukuran populasi GA
    - generations: jumlah generasi evolusi
    - crossover_prob (cxpb): probabilitas crossover
    - mutation_prob (mutpb): probabilitas mutasi

    Return:
    - best_individual: individu terbaik (list [RSI_buy, RSI_sell])
    """

    # ==== 1. Definisikan Rentang (Bound) Untuk Parameter RSI ====
    RSI_MIN = 10
    RSI_MAX = 90

    # ==== 2. Definisikan Fungsi Fitness ====
    def fitness(individual):
        """
        Menghitung akurasi strategi berdasarkan sinyal RSI.
        - individual: [RSI_buy, RSI_sell]
        """
        RSI_buy, RSI_sell = individual

        # Buat prediksi dari data X
        # Asumsi kolom RSI berada di indeks tertentu (misal X[:, 0] atau X[:, 6])
        # Di sini, kita contohkan RSI berada di X[:, 0]
        predictions = []
        for rsi in X[:, 0]:  
            if rsi < RSI_buy:
                predictions.append(1)  # sinyal beli
            elif rsi > RSI_sell:
                predictions.append(0)  # sinyal jual
            else:
                predictions.append(-1) # tidak ada aksi

        # Hanya hitung akurasi pada prediksi != -1
        valid = y[np.array(predictions) != -1]
        pred = np.array(predictions)[np.array(predictions) != -1]

        if len(valid) == 0:
            return 0.0,

        accuracy = np.sum(valid == pred) / len(valid)
        return accuracy,

    # ==== 3. Setup DEAP (Inisialisasi) ====

    # a. Definisikan Tipe Fitness dan Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maksimalkan fitness
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # b. Definisikan Atribut
    toolbox.register("attr_RSI_buy", random.uniform, RSI_MIN, RSI_MAX)
    toolbox.register("attr_RSI_sell", random.uniform, RSI_MIN, RSI_MAX)

    # c. Definisikan Individual & Populasi
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_RSI_buy, toolbox.attr_RSI_sell),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ==== 4. Definisikan Fungsi Mutasi Uniform Kustom ====

    def mutUniformFloat(individual, low, up, indpb):
        """
        Mutasi uniform untuk atribut float dalam range [low, up] dengan probabilitas indpb.
        :param individual: list individu (misal [RSI_buy, RSI_sell])
        :param low: batas bawah float atau list floats
        :param up: batas atas float atau list floats
        :param indpb: probabilitas mutasi per gene
        """
        size = len(individual)
        # Jika low & up adalah float tunggal, ubah ke list agar seukuran 'size'
        if not hasattr(low, '__iter__'):
            low = [low] * size
        if not hasattr(up, '__iter__'):
            up = [up] * size

        for i, (xl, xu) in enumerate(zip(low, up)):
            if random.random() < indpb:
                individual[i] = random.uniform(xl, xu)
        return individual,

    # ==== 5. Registrasi Operator GA ====

    # a. Fitness Function
    toolbox.register("evaluate", fitness)

    # b. Crossover
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    # c. Mutasi (menggunakan fungsi kustom di atas)
    toolbox.register(
        "mutate",
        mutUniformFloat,
        low=RSI_MIN,
        up=RSI_MAX,
        indpb=0.2
    )

    # d. Seleksi
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ==== 6. Buat Populasi Awal ====
    population = toolbox.population(n=population_size)

    # ==== 7. Statistik & Hall of Fame ====
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # ==== 8. Jalankan Evolusi (eaSimple) ====
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=generations,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True
    )

    # ==== 9. Keluarkan Hasil Terbaik ====
    best_individual = hall_of_fame[0]
    print(f"Best individual: RSI_buy={best_individual[0]:.2f}, RSI_sell={best_individual[1]:.2f}")
    return best_individual

# ============ Contoh Penggunaan ============ #
if __name__ == "__main__":
    # Contoh dataset sederhana:
    # Misalkan X adalah array 2D, kolom pertama adalah RSI, kolom kedua dll.
    # y adalah label biner (0 = jual benar, 1 = beli benar)
    import numpy as np
    
    # Contoh data RSI acak (50 baris, 1 kolom RSI)
    X_dummy = np.random.uniform(low=0, high=100, size=(50, 1))
    # Contoh target acak
    y_dummy = np.random.randint(0, 2, size=(50,))

    # Panggil fungsi genetic_algorithm
    best = genetic_algorithm(
        X_dummy,
        y_dummy,
        population_size=20,
        generations=10,
        crossover_prob=0.5,
        mutation_prob=0.2
    )
    print("Best Result:", best)
