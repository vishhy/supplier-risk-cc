# Supplier Risk Command Centre

An interactive Streamlit dashboard for the Strategic Supplier Churn & Risk Mitigation
study: descriptive spend and performance analytics, an adjustable supplier scorecard,
out-of-sample failure prediction, and a what-if simulator.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

Opens at http://localhost:8501.

## Deploying to Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. At share.streamlit.io, click **New app** and point it at the repo.
3. Set **Main file path** to `main.py`.
4. Deploy. The build reads `requirements.txt` automatically.

The whole data layer is 0.7 MB of Parquet plus a 3.5 MB model file, so the app
starts in a few seconds and stays well inside the free tier's memory limit.
Nothing is trained at runtime: `@st.cache_data` holds the tables and
`@st.cache_resource` holds the fitted pipeline.

## How it is put together

```
main.py                  page config, the global filter rail, navigation
lib/data.py              cached loaders, the cross-filter, scorecard and cost maths
lib/theme.py             navy/gold chrome, chart template, KPI cards
views/                   one file per page
data/*.parquet           pre-aggregated tables
models/rf_pipeline.joblib  fitted Random Forest pipeline
prepare.py               rebuilds everything in data/ and models/ from the raw CSVs
```

To regenerate the data layer after changing the source CSVs:

```bash
python prepare.py
```

## The pages

**Executive Overview** — spend concentration with ABC classes, delivery reliability
over time with the festive-quarter effect marked, category performance, and the
current risk profile weighted by spend.

**Delivery & Invoicing** — the descriptive layer across three tabs: delivery
reliability by category and quarter, invoice accuracy and dispute behaviour, and
SLA compliance against what each supplier actually contracted to.

**Supplier Scorecard** — four pillar weight sliders that re-rank all suppliers
instantly. The page reports rank correlation and bottom-30 retention against the
40/20/15/25 base so you can see how much the conclusion depends on the weighting.

**Risk Prediction** — out-of-sample predictions with a model selector, an
intervention threshold driven by your own cost assumptions, a cost curve, a
confusion matrix, ROC comparison, capacity-constrained triage, and a downloadable
watchlist.

**What-If Simulator** — pick a supplier, move any driver, and the model re-scores
live. A sensitivity sweep shows which lever moves that specific supplier most.

**Portfolio Strategy** — Kraljic positioning with adjustable quadrant cut-offs and
a recommended posture per quadrant.

**Supplier 360** — one supplier's full record: delivery history, financial
trajectory, incident log and percentile position against category peers.

## Global filters

Every page reads the same filter state from the sidebar: quarter range, category,
tier, region, business size, ABC class and sourcing status. Changing a filter
re-computes every chart, table and model output on every page, and the active
selection is shown as chips at the top of each page.

## An honest note carried through the app

The Random Forest reaches 0.890 out-of-sample AUC. Ranking suppliers by their
two-quarter rolling on-time rate alone reaches 0.878, and that difference is not
statistically significant across folds. The model selector on the Risk Prediction
page lets you switch between them and see how little changes. Most of the value
here comes from ranking systematically at all rather than from the algorithm,
which is the study's main finding and not something the dashboard hides.
