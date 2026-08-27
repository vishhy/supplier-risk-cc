"""Cached data access and the global cross-filter, applied consistently on every page."""
import streamlit as st
import pandas as pd, numpy as np, json, joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'data'
M = ROOT / 'models'

# --------------------------------------------------------------------- loaders
@st.cache_data(show_spinner=False)
def meta():
    return json.loads((D/'meta.json').read_text())

@st.cache_data(show_spinner=False)
def model_metrics():
    return json.loads((D/'model_metrics.json').read_text())

@st.cache_data(show_spinner=False)
def suppliers():      return pd.read_parquet(D/'supplier_summary.parquet')
@st.cache_data(show_spinner=False)
def supplier_quarter():return pd.read_parquet(D/'supplier_quarter.parquet')
@st.cache_data(show_spinner=False)
def category_quarter():return pd.read_parquet(D/'category_quarter.parquet')
@st.cache_data(show_spinner=False)
def events():         return pd.read_parquet(D/'events.parquet')
@st.cache_data(show_spinner=False)
def contracts():      return pd.read_parquet(D/'contracts.parquet')
@st.cache_data(show_spinner=False)
def invoice_errors(): return pd.read_parquet(D/'invoice_errors.parquet')
@st.cache_data(show_spinner=False)
def feature_importance(): return pd.read_parquet(D/'feature_importance.parquet')

@st.cache_data(show_spinner=False)
def scored():
    s = pd.read_parquet(D/'scored_panel.parquet')
    q = meta()['quarters']; qi = {k:i for i,k in enumerate(q)}
    s['qi'] = s.fiscal_quarter.map(qi)
    return s

@st.cache_resource(show_spinner=False)
def model():
    return joblib.load(M/'rf_pipeline.joblib')

QUARTERS = None
def quarters():
    global QUARTERS
    if QUARTERS is None: QUARTERS = meta()['quarters']
    return QUARTERS

def scored_quarters():
    """Quarters that carry an out-of-sample prediction."""
    return model_metrics()['quarters_scored']

# --------------------------------------------------------------------- filters
FILTER_KEYS = ['f_cat','f_tier','f_region','f_size','f_abc','f_sole','f_q']

def init_filters():
    s = suppliers()
    st.session_state.setdefault('f_cat',   [])
    st.session_state.setdefault('f_tier',  [])
    st.session_state.setdefault('f_region',[])
    st.session_state.setdefault('f_size',  [])
    st.session_state.setdefault('f_abc',   [])
    st.session_state.setdefault('f_sole',  'All suppliers')
    # f_q is seeded by the widget itself (value=), not here: a select_slider given a
    # pre-seeded scalar in session_state renders as a single-value slider.

def reset_filters():
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)
    init_filters()

def active_filters():
    """Human-readable list of what is currently narrowing the view."""
    out = []
    ss = st.session_state
    for key, label in [('f_cat','Category'),('f_tier','Tier'),('f_region','Region'),
                       ('f_size','Size'),('f_abc','ABC')]:
        v = ss.get(key) or []
        if v: out.append(f'{label}: ' + ', '.join(map(str, v)))
    if ss.get('f_sole','All suppliers') != 'All suppliers':
        out.append(ss['f_sole'])
    lo, hi = quarter_range()
    qs = quarters()
    if lo != 0 or hi != len(qs)-1:
        out.append(f'{qs[lo]} to {qs[hi]}')
    return out

def supplier_ids():
    """The supplier set surviving every active attribute filter. This is the join key
    that makes filtering behave consistently across every page."""
    s = suppliers()
    ss = st.session_state
    m = pd.Series(True, index=s.index)
    if ss.get('f_cat'):    m &= s.primary_category.isin(ss['f_cat'])
    if ss.get('f_tier'):   m &= s.supplier_tier.isin(ss['f_tier'])
    if ss.get('f_region'): m &= s.region.isin(ss['f_region'])
    if ss.get('f_size'):   m &= s.business_size.isin(ss['f_size'])
    if ss.get('f_abc'):    m &= s.abc_class.isin(ss['f_abc'])
    sole = ss.get('f_sole','All suppliers')
    if sole == 'Sole-sourced only':        m &= s.is_sole_source == 1
    elif sole == 'Has alternatives only':  m &= s.is_sole_source == 0
    return set(s.loc[m, 'supplier_id'])

def quarter_range():
    """Index bounds of the selected quarter range. Defensive: the widget can hand back a
    scalar before it has been interacted with, so fall back to the full range."""
    q = quarters()
    v = st.session_state.get('f_q')
    if isinstance(v, (tuple, list)) and len(v) == 2:
        lo, hi = v
    else:
        lo, hi = q[0], q[-1]
    try:
        a, b = q.index(lo), q.index(hi)
    except ValueError:
        return 0, len(q)-1
    return (a, b) if a <= b else (b, a)

def fs():
    """Filtered supplier summary."""
    ids = supplier_ids()
    return suppliers()[suppliers().supplier_id.isin(ids)].copy()

def fsq():
    """Filtered supplier-quarter table, respecting both attribute and time filters."""
    ids = supplier_ids(); lo, hi = quarter_range()
    d = supplier_quarter()
    return d[d.supplier_id.isin(ids) & d.qi.between(lo, hi)].copy()

def fscored(only_scored_quarters=True):
    """Filtered modelling panel with out-of-sample predictions attached."""
    ids = supplier_ids(); lo, hi = quarter_range()
    d = scored()
    d = d[d.supplier_id.isin(ids) & d.qi.between(lo, hi)]
    if only_scored_quarters:
        d = d[d.prob_rf.notna()]
    return d.copy()

def fevents():
    ids = supplier_ids(); lo, hi = quarter_range()
    q = quarters(); qi = {k:i for i,k in enumerate(q)}
    d = events().copy()
    d['qi'] = d.fiscal_quarter.map(qi)
    return d[d.supplier_id.isin(ids) & d.qi.between(lo, hi)].copy()

def latest_scored_quarter():
    return scored_quarters()[-1]

# --------------------------------------------------------------------- scorecard
def pct_rank(s, higher_is_better=True):
    r = s.rank(pct=True)
    return (r if higher_is_better else 1 - r) * 100

def build_scorecard(df, w_delivery, w_quality, w_invoice, w_financial):
    """Recompute the composite score for whatever supplier set and weights are supplied.
    Weights are normalised so the score always sits on 0-100 regardless of slider totals."""
    d = df.copy()
    tot = max(w_delivery + w_quality + w_invoice + w_financial, 1e-9)
    wd, wq, wi, wf = (w_delivery/tot, w_quality/tot, w_invoice/tot, w_financial/tot)
    d['delivery_score']  = pct_rank(d.otif_rate, True)*0.6 + pct_rank(d.avg_delay_days, False)*0.4
    d['quality_score']   = pct_rank(d.reject_rate_pct, False)
    d['invoice_score']   = pct_rank(d.invoice_accuracy, True)*0.7 + pct_rank(d.dispute_rate, False)*0.3
    d['financial_score'] = pct_rank(d.external_credit_score, True)
    d['composite_score'] = (d.delivery_score*wd + d.quality_score*wq +
                            d.invoice_score*wi + d.financial_score*wf)
    d['grade'] = pd.cut(d.composite_score, [-0.01,25,50,75,100.01],
                        labels=['D — Exit / Replace','C — Develop','B — Maintain','A — Preferred'])
    d = d.sort_values('composite_score', ascending=False).reset_index(drop=True)
    d['rank'] = np.arange(1, len(d)+1)
    return d

BASE_WEIGHTS = dict(w_delivery=40, w_quality=20, w_invoice=15, w_financial=25)

# --------------------------------------------------------------------- economics
def cost_table(prob, y, cost_miss, cost_action, grid=None):
    if grid is None: grid = np.arange(0.02, 0.99, 0.02)
    prob = np.asarray(prob); y = np.asarray(y)
    rows = []
    for t in grid:
        pred = prob >= t
        tp = int((pred & (y==1)).sum()); fp = int((pred & (y==0)).sum())
        fn = int((~pred & (y==1)).sum()); tn = int((~pred & (y==0)).sum())
        rows.append(dict(threshold=round(float(t),2), flagged=tp+fp, tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),
                         cost=fn*cost_miss + (tp+fp)*cost_action))
    return pd.DataFrame(rows)

def risk_band(p):
    if p >= 0.70: return 'Critical'
    if p >= 0.45: return 'High'
    if p >= 0.20: return 'Moderate'
    return 'Low'
