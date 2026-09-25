import numpy as np

# ---------------- FITNESS FUNCTION ----------------
def predict_time(model, sol):
    sol = np.array(sol).reshape(1, -1)
    return model.predict(sol)[0]


# ---------------- GENETIC ALGORITHM ----------------
def run_ga(model, generations=30, population_size=15):

    best_solution = None
    best_score = float("inf")

    # ---------------- INITIAL POPULATION ----------------
    population = []

    for _ in range(population_size):
        # Match PSO search-space bounds so GA and PSO optimize the same variables/ranges.
        # Order: [MapperRAM, ReducerRAM, Reducers, BlockSize, Compression, CPUUsage, MemoryUsage]
        individual = [
            np.random.randint(8, 17),   # 8..16
            np.random.randint(8, 17),   # 8..16
            np.random.randint(15, 51),  # 15..50
            np.random.randint(120, 257),# 120..256
            np.random.randint(0, 2),    # 0..1
            np.random.randint(40, 101), # 40..100
            np.random.randint(40, 101), # 40..100
        ]

        population.append(individual)

    # ---------------- EVOLUTION LOOP ----------------
    for _ in range(generations):

        new_population = []

        for i in range(population_size):

            # SELECT TWO RANDOM PARENTS
            p1 = population[np.random.randint(0, population_size)]
            p2 = population[np.random.randint(0, population_size)]

            # ---------------- CROSSOVER ----------------
            child = []
            for j in range(len(p1)):
                if np.random.rand() > 0.5:
                    child.append(p1[j])
                else:
                    child.append(p2[j])

            # ---------------- MUTATION ----------------
            mutation_index = np.random.randint(0, 7)
            if mutation_index == 0:
                child[0] = np.random.randint(8, 17)   # 8..16
            elif mutation_index == 1:
                child[1] = np.random.randint(8, 17)   # 8..16
            elif mutation_index == 2:
                child[2] = np.random.randint(15, 51)  # 15..50
            elif mutation_index == 3:
                child[3] = np.random.randint(120, 257) # 120..256
            elif mutation_index == 4:
                child[4] = np.random.randint(0, 2)    # 0..1
            elif mutation_index == 5:
                child[5] = np.random.randint(40, 101) # 40..100
            else:
                child[6] = np.random.randint(40, 101) # 40..100


            new_population.append(child)

        population = new_population

        # ---------------- EVALUATE POPULATION ----------------
        for individual in population:
            score = predict_time(model, individual)

            if score < best_score:
                best_score = score
                best_solution = individual

    return best_solution, best_score