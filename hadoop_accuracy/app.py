import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

from ga import run_ga
from pso import run_pso

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hadoop Optimization", layout="wide")

# ---------------- SESSION STATE ----------------
if "data" not in st.session_state:
    st.session_state.data = None

if "X_train" not in st.session_state:
    st.session_state.X_train = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_train" not in st.session_state:
    st.session_state.y_train = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "model_gbrt" not in st.session_state:
    st.session_state.model_gbrt = None

if "model_xgb" not in st.session_state:
    st.session_state.model_xgb = None


# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", [
    "Home",
    "Dataset Generator",
    "Upload Dataset",
    "Preprocessing",
    "Training",
    "Prediction",
    "GA Optimization",
    "PSO Optimization",
    "Comparison"
])

# ---------------- HOME ----------------
if page == "Home":
    st.title("Adaptive Hadoop Configuration Optimization")
    st.info("XGBoost + PSO vs GBRT + GA Framework")
    st.markdown(
        """
        ### How to use this demo
        1. **Dataset Generator / Upload Dataset**
        2. **Preprocessing**
        3. **Training** (GBRT and/or XGBoost)
        4. **GA Optimization** and **PSO Optimization**
        5. **Comparison** (runs several times and shows aggregated results)

        **Goal:** find Hadoop configuration parameters that minimize the predicted **ExecutionTime**.
        """
    )



# ---------------- DATASET GENERATOR ----------------
elif page == "Dataset Generator":
    st.title("Dataset Generator")

    from dataset_generator import generate_dataset

    n = st.slider("Dataset size (rows)", min_value=500, max_value=10000, value=3000, step=500)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

    if st.button("Generate Dataset"):
        # dataset_generator currently uses a fixed seed internally; we re-seed numpy here to keep demo consistent
        np.random.seed(int(seed))
        df = generate_dataset(n=int(n))
        st.session_state.data = df
        st.success("Dataset Generated!")

    if st.session_state.data is not None:
        st.write(st.session_state.data.head())

        csv = st.session_state.data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            csv,
            "hadoop_dataset.csv",
            "text/csv"
        )



# ---------------- UPLOAD ----------------
elif page == "Upload Dataset":
    st.title("Upload Dataset")

    file = st.file_uploader("Upload CSV")

    if file:
        st.session_state.data = pd.read_csv(file)
        st.success("Dataset Loaded!")

    if st.session_state.data is not None:
        st.write(st.session_state.data.head())


# ---------------- PREPROCESSING ----------------
elif page == "Preprocessing":
    st.title("Preprocessing")

    if st.session_state.data is None:
        st.error("Please generate or upload dataset first!")

    else:
        from preprocess import preprocess

        df, report = preprocess(st.session_state.data)

        st.session_state.data = df

        st.success("Preprocessing Completed!")

        # ---------------- SHOW REPORT ----------------
        st.subheader("📊 Preprocessing Report")

        st.write("Original Shape:", report["original_shape"])
        st.write("Duplicates Removed:", report["duplicates_removed"])
        st.write("Null Rows Removed:", report["null_rows_removed"])
        st.write("Outliers Handled:", report["outliers_handled"])
        st.write("Final Shape:", report["final_shape"])

        # ---------------- SHOW DATA ----------------
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df.head())


# ---------------- TRAINING ----------------
elif page == "Training":
    st.title("Train Models")


    if st.session_state.data is None:
        st.error("Dataset not available!")
    else:
        # Always preprocess before training to ensure consistency with the demo requirements.
        from preprocess import preprocess

        df, report = preprocess(st.session_state.data)
        st.session_state.data = df

        X = df.drop("ExecutionTime", axis=1)
        y = df["ExecutionTime"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        # ---------------- GBRT ----------------
        if st.button("Train GBRT"):
            model = GradientBoostingRegressor()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            st.session_state.model_gbrt = model

            st.success("GBRT Trained")
            st.write("MAE:", mean_absolute_error(y_test, pred))
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
            st.write("R2:", r2_score(y_test, pred))

        # ---------------- XGBOOST ----------------
        if st.button("Train XGBoost"):
            # Tuned hyperparameters to reliably achieve high R2 on the synthetic non-linear dataset.
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=1.0,
                min_child_weight=1,
                random_state=42
            )


            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            st.session_state.model_xgb = model

            st.success("XGBoost Trained")
            st.write("MAE:", mean_absolute_error(y_test, pred))
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
            st.write("R2:", r2_score(y_test, pred))
            st.write("Model R2 Score:", r2_score(y_test, pred))

# ---------------- PREDICTION ----------------
elif page == "Prediction":
    st.title("Execution Time Prediction")

    if st.session_state.model_xgb is None:
        st.warning("Train XGBoost first!")
    else:
        m1 = st.slider("MapperRAM", 1, 16, 4)
        m2 = st.slider("ReducerRAM", 1, 16, 4)
        r = st.slider("Reducers", 1, 50, 10)
        b = st.slider("BlockSize", 64, 256, 128)
        c = st.selectbox("Compression", [0, 1])
        cpu = st.slider("CPUUsage", 10, 100, 50)
        mem = st.slider("MemoryUsage", 10, 100, 50)

        if st.button("Predict Execution Time"):
            input_data = np.array([[m1, m2, r, b, c, cpu, mem]])

            pred = st.session_state.model_xgb.predict(input_data)

            st.success(f"Predicted Execution Time: {pred[0]:.2f} sec")


# ---------------- GA OPTIMIZATION ----------------
elif page == "GA Optimization":
    st.title("Genetic Algorithm Optimization")

    runs = st.slider("Runs (averaging)", min_value=1, max_value=20, value=5)


    if st.session_state.model_gbrt is None:
        st.error("Train GBRT first!")
    else:
        # GA baseline uses GBRT predictions as the fitness function.
        scores = []
        best_solutions = []
        for _ in range(int(runs)):
            best, score = run_ga(st.session_state.model_gbrt)
            scores.append(float(score))
            best_solutions.append(best)

        best_idx = int(np.argmin(scores))
        best = best_solutions[best_idx]
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        st.success(f"Best Solution (best of runs): {best}")
        st.info(f"ExecutionTime mean±std over runs (GBRT): {mean_score:.3f} ± {std_score:.3f}")




# ---------------- PSO OPTIMIZATION ----------------
elif page == "PSO Optimization":
    st.title("Particle Swarm Optimization")

    runs = st.slider("Runs (averaging)", min_value=1, max_value=20, value=5)


    if st.session_state.model_xgb is None:
        st.error("Train XGBoost first!")
    else:
        scores = []
        best_solutions = []
        for _ in range(int(runs)):
            best, score = run_pso(st.session_state.model_xgb)
            scores.append(float(score))
            best_solutions.append(best)

        best_idx = int(np.argmin(scores))
        best = best_solutions[best_idx]
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        st.success(f"Best Solution (best of runs): {best}")
        st.info(f"ExecutionTime mean±std over runs (XGBoost): {mean_score:.3f} ± {std_score:.3f}")



# ---------------- COMPARISON ----------------
elif page == "Comparison":

    st.title("GA vs PSO Comparison (Real ML Based)")

    runs = st.slider("Runs (averaging)", min_value=1, max_value=20, value=5)


    if st.session_state.model_xgb is None:
        st.error("Train model first!")
    else:
        model_gbrt = st.session_state.model_gbrt
        model_xgb = st.session_state.model_xgb

        # Compare apples-to-apples by evaluating the *same candidate solutions* under BOTH models.
        # This shows whether PSO is better under its own surrogate (XGB) and/or under GBRT.
        ga_scores_gbrt = []
        pso_scores_gbrt = []
        ga_scores_xgb = []
        pso_scores_xgb = []

        for _ in range(int(runs)):
            ga_best, _ga_score = run_ga(model_gbrt)   # GA searches with GBRT fitness
            pso_best, _pso_score = run_pso(model_xgb) # PSO searches with XGB fitness

            ga_vec = np.array(ga_best).reshape(1, -1)
            pso_vec = np.array(pso_best).reshape(1, -1)

            # Evaluate GA & PSO solutions with both predictors
            ga_gbrt = float(model_gbrt.predict(ga_vec)[0])
            pso_gbrt = float(model_gbrt.predict(pso_vec)[0])
            ga_xgb = float(model_xgb.predict(ga_vec)[0])
            pso_xgb = float(model_xgb.predict(pso_vec)[0])

            ga_scores_gbrt.append(ga_gbrt)
            pso_scores_gbrt.append(pso_gbrt)
            ga_scores_xgb.append(ga_xgb)
            pso_scores_xgb.append(pso_xgb)

        def summarize(scores):
            scores = np.array(scores, dtype=float)
            return {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "best": float(np.min(scores)),
            }

        gbrt_g = summarize(ga_scores_gbrt)
        gbrt_p = summarize(pso_scores_gbrt)
        xgb_g = summarize(ga_scores_xgb)
        xgb_p = summarize(pso_scores_xgb)

        #st.markdown("## Evaluation with GBRT (GBRT predictions)")
        df_gbrt = pd.DataFrame({

            "Method": ["GBRT+GA", "XGBoost+PSO"],
            "MeanExecutionTime": [gbrt_g["mean"], gbrt_p["mean"]],
            "StdExecutionTime": [gbrt_g["std"], gbrt_p["std"]],
            "BestExecutionTime": [gbrt_g["best"], gbrt_p["best"]],
        })
        fig_gbrt = px.bar(
            df_gbrt,
            x="Method",
            y="MeanExecutionTime",
            color="Method",
            error_y="StdExecutionTime",
            labels={"MeanExecutionTime": "Mean Predicted Execution Time (GBRT)"}
        )
        st.plotly_chart(fig_gbrt)

        #st.subheader("Evaluation with XGBoost (XGB predictions)")
        df_xgb = pd.DataFrame({
            "Method": ["GBRT+GA", "XGBoost+PSO"],
            "MeanExecutionTime": [xgb_g["mean"], xgb_p["mean"]],
            "StdExecutionTime": [xgb_g["std"], xgb_p["std"]],
            "BestExecutionTime": [xgb_g["best"], xgb_p["best"]],
        })
        fig_xgb = px.bar(
            df_xgb,
            x="Method",
            y="MeanExecutionTime",
            color="Method",
            error_y="StdExecutionTime",
            labels={"MeanExecutionTime": "Mean Predicted Execution Time (XGB)"}
        )
        st.plotly_chart(fig_xgb)

        st.success(
            "Comparison finished (showing both-surrogate evaluation). "
            f"GBRT: GA {gbrt_g['mean']:.3f}±{gbrt_g['std']:.3f} | PSO {gbrt_p['mean']:.3f}±{gbrt_p['std']:.3f}. "
            f"XGB: GA {xgb_g['mean']:.3f}±{xgb_g['std']:.3f} | PSO {xgb_p['mean']:.3f}±{xgb_p['std']:.3f}."
        )

