# MP-OPT Interactive Research Lab

A browser-based, **educational** Streamlit application explaining the
two-stage variable-length time-series anomaly detection framework from
Sutrisno & Phoa (2025).

> This is not the paper authors' implementation or a validated reproduction
> of the published results. All interactive time-series data are synthetic.

## Live features

1. **Dataset** — generate periodic traffic-flow-like series with a controlled anomaly.
2. **Stage 1: MP + STOMP** — change the fixed window, inspect Matrix Profile,
   nearest neighbors, and Top-k non-overlapping candidates.
3. **Stage 2: LHS + score** — explore starting positions and variable lengths,
   comparing them against sampled reference patterns.
4. **Stage 2: HDBSCAN + CSOS-inspired search** — run a teaching optimizer,
   inspect convergence and agent locations. HDBSCAN uses scikit-learn; the
   CSOS mechanics remain simplified and are labeled accordingly.
5. **Spatial-temporal demonstration** — inspect delayed downstream patterns
   with DTW and Pearson correlation.
6. **Paper results** — distinguish the published tables from simulated data.

## Deploy with Streamlit Community Cloud

1. Create a **separate repository** such as `mp-opt-interactive-lab` in GitHub.
   This avoids modifying the professor's protected `2026lance` meeting-site
   build workflow.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
   Upload `.streamlit/config.toml` into the `.streamlit` folder.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub. Choose **Create app**, select the repository and the `main` branch,
   then set the entrypoint to `app.py`.
4. Deploy. Once it becomes available, copy its public `*.streamlit.app` URL
   into your next `.rst` meeting report.

### Local launch

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Technical and scientific cautions

- STOMP is a dot-product recurrence in a small univariate educational setting;
  not benchmarked for million-point production data.
- LHS is a 1-D stratified sampling example, not the complete paper setup.
- CSOS is an inspired teaching implementation with initially fixed groups,
  bounded integer solutions, and cached objective evaluations.
- Correlation of simulated delayed patterns does not prove causation.
- The published accuracy table uses relative anomaly scores, **not**
  supervised classification accuracy.
- Avoid presenting the synthetic results as measurements from Taiwan freeway
  detectors.
