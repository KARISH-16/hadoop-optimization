import numpy as np

# ---------------- FITNESS FUNCTION ----------------
def predict_time(model, sol):
    sol = np.array(sol).reshape(1, -1)
    return model.predict(sol)[0]


# ---------------- PSO ALGORITHM ----------------
def run_pso(model, particles=20, iterations=40, no_improvement_iters=10, stagnation_tol=1e-9):

    # ---------------- SEARCH SPACE LIMITS ----------------
    lb = np.array([8, 8, 15, 120, 0, 40, 40])
    ub = np.array([16, 16, 50, 256, 1, 100, 100])

    dim = 7

    # ---------------- INITIALIZATION ----------------
    X = np.random.uniform(lb, ub, (particles, dim))
    V = np.zeros((particles, dim))

    pbest = X.copy()
    pbest_score = np.array([float("inf")] * particles)

    gbest = None
    gbest_score = float("inf")

    w = 0.7   # inertia
    c1 = 1.5  # personal best
    c2 = 1.5  # global best

    # ---------------- ITERATIONS (with early stop) ----------------
    no_improve_count = 0
    last_best = gbest_score

    for _ in range(iterations):

        improved_this_iter = False

        for i in range(particles):

            sol = X[i].astype(int)

            score = predict_time(model, sol)

            # update personal best
            if score < pbest_score[i]:
                pbest_score[i] = score
                pbest[i] = X[i].copy()

            # update global best
            if score < gbest_score:
                gbest_score = score
                gbest = X[i].copy()
                improved_this_iter = True

        # early stopping check
        if improved_this_iter:
            no_improve_count = 0
            last_best = gbest_score
        else:
            no_improve_count += 1
            if abs(last_best - gbest_score) <= stagnation_tol:
                if no_improve_count >= no_improvement_iters:
                    break

        # ---------------- UPDATE VELOCITY & POSITION ----------------
        for i in range(particles):

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            V[i] = (
                w * V[i]
                + c1 * r1 * (pbest[i] - X[i])
                + c2 * r2 * (gbest - X[i])
            )
            
            # small decay to help convergence stability
            V[i] *= 0.99

            X[i] = X[i] + V[i]

            # ---------------- BOUNDARY CHECK ----------------
            X[i] = np.clip(X[i], lb, ub)

    return gbest.astype(int).tolist(), gbest_score