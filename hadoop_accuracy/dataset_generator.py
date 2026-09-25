import numpy as np
import pandas as pd

def generate_dataset(n=5000):
    np.random.seed(42)

    df = pd.DataFrame({
        "MapperRAM": np.random.randint(1, 16, n),
        "ReducerRAM": np.random.randint(1, 16, n),
        "Reducers": np.random.randint(1, 50, n),
        "BlockSize": np.random.randint(64, 256, n),
        "Compression": np.random.randint(0, 2, n),
        "CPUUsage": np.random.randint(10, 100, n),
        "MemoryUsage": np.random.randint(10, 100, n),
    })

    df["ExecutionTime"] = (
        (df["MapperRAM"] * df["Reducers"]) * 0.8 +
        (df["ReducerRAM"] ** 1.3) +
        (df["BlockSize"] / 10) +
        (100 / (df["CPUUsage"] + 1)) +
        (df["MemoryUsage"] * np.log(df["Reducers"] + 1)) +
        (50 * (1 - df["Compression"])) +
        np.random.normal(0, 3, n)
    )

    return df