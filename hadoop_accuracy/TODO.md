# TODO - Final Year Project Demo (Hadoop Optimization)

- [ ] Refactor Streamlit dataset generation to use `dataset_generator.generate_dataset()` (non-linear) instead of inline linear-ish formula.
- [ ] Refactor Training page to use `preprocess.preprocess()` consistently and display preprocessing report.
- [ ] Implement robust model training:
  - [ ] Train GBRT baseline and evaluate (MAE, RMSE, R2).
  - [ ] Train XGBoost with tuned hyperparameters (target R2 > 0.85).
  - [ ] Store both models in `st.session_state`.
- [ ] Unify search-space constraints between GA and PSO via `utils.py`.
- [ ] Implement fair optimization:
  - [ ] GA uses GBRT prediction as fitness.
  - [ ] PSO uses XGBoost prediction as fitness.
- [ ] Add multi-run averaging for GA and PSO (mean/std) and ensure comparison uses these aggregates.
- [ ] Update Comparison page:
  - [ ] Compute GBRT+GA vs XGBoost+PSO (mean execution time, std, best).
  - [ ] Render bar chart + metrics.
- [ ] Add convergence/best-so-far trace summaries if available from GA/PSO, else show best per iteration/run.
- [ ] Validate by running `streamlit run app.py` and ensure R2 > 0.85 for XGBoost and PSO outperforms GA on execution time (averaged over runs).

