import streamlit as st
import pandas as pd, numpy as np
import plotly.graph_objects as go
from lib import data as D
from lib.theme import *

st.title('What-If Simulator')
st.markdown('<div class="sub">Pick a supplier, change what you believe about them, and the '
            'model re-scores immediately. Use it to ask what would have to improve for a '
            'supplier to fall out of the risk band.</div>', unsafe_allow_html=True)

sc = D.fscored()
if sc.empty:
    st.warning('No scored supplier-quarters in this selection.'); st.stop()

bundle = D.model()
pipe, NUM, CAT = bundle['pipeline'], bundle['num'], bundle['cat']

lq = D.latest_scored_quarter()
avail = sc[sc.fiscal_quarter == lq]
if avail.empty:
    lq = sc.sort_values('qi').fiscal_quarter.iloc[-1]
    avail = sc[sc.fiscal_quarter == lq]

pick = st.columns([2.2, 1, 1])
names = avail.sort_values('prob_rf', ascending=False)
label = names.supplier_name + '  ·  p=' + names.prob_rf.round(2).astype(str)
sel = pick[0].selectbox(f'Supplier (scored on {lq})', options=list(names.supplier_id),
                        format_func=lambda i: label[names.supplier_id == i].iloc[0],
                        key='sim_supplier')
row = avail[avail.supplier_id == sel].iloc[0]
pick[1].markdown(kpi('Tier', row.supplier_tier, row.primary_category), unsafe_allow_html=True)
pick[2].markdown(kpi('Spend share', f'{row.spend_share_of_quarter_pct:.2f}%',
                     'of the quarter'), unsafe_allow_html=True)

BASE = pd.DataFrame([row[NUM+CAT]])
base_p = float(pipe.predict_proba(BASE)[:,1][0])

# ------------------------------------------------------------------ levers
# scale = multiplier applied to the slider value before it reaches the model, so
# every rate can be shown as a percentage rather than a bare fraction
LEVERS = [
    ('rolling_2q_on_time_rate', 'Two-quarter rolling on-time rate (%)', 0.0, 100.0, 1.0, 0.01),
    ('on_time_rate',            'On-time rate this quarter (%)',        0.0, 100.0, 1.0, 0.01),
    ('otif_rate',               'OTIF rate (%)',                        0.0, 100.0, 1.0, 0.01),
    ('avg_delay_days',          'Average delay (days)',                 0.0, 30.0, 0.5, 1.0),
    ('reject_rate_pct',         'Rejection rate (%)',                   0.0, 10.0, 0.1, 1.0),
    ('invoice_error_rate',      'Invoice error rate (%)',               0.0, 100.0, 1.0, 0.01),
    ('external_credit_score',   'External credit score',                1.0, 100.0, 1.0, 1.0),
    ('altman_z_score',          'Altman Z score',                      -1.0, 7.0, 0.1, 1.0),
    ('days_sales_outstanding',  'Days sales outstanding',               10.0, 220.0, 5.0, 1.0),
    ('promoter_pledge_pct',     'Promoter pledge (%)',                  0.0, 95.0, 1.0, 1.0),
    ('event_count',             'Incidents logged this quarter',        0.0, 12.0, 1.0, 1.0),
    ('po_count_qoq_growth_pct', 'Order volume change QoQ (%)',        -80.0, 200.0, 5.0, 1.0),
]
LEVERS = [l for l in LEVERS if l[0] in NUM]

st.markdown('##### Adjust the assumptions')
if st.button('Reset to actual values'):
    for k,*_ in LEVERS: st.session_state.pop('sim_'+k, None)
    st.rerun()

sim = BASE.copy()
cols = st.columns(3)
for i, (key, lab, lo, hi, step, scale) in enumerate(LEVERS):
    raw = row[key]
    raw = float(raw) if pd.notna(raw) else (lo+hi)/2*scale
    cur = float(np.clip(raw/scale, lo, hi))
    with cols[i % 3]:
        v = st.slider(lab, lo, hi, round(cur, 2), step, key='sim_'+key)
        sim.loc[:, key] = v*scale

new_p = float(pipe.predict_proba(sim)[:,1][0])
delta = new_p - base_p

# ------------------------------------------------------------------ result
st.write('')
r = st.columns([1,1,1,1.6])
r[0].markdown(kpi('Actual probability', f'{base_p:.1%}', f'as scored in {lq}'),
              unsafe_allow_html=True)
r[1].markdown(kpi('Simulated probability', f'{new_p:.1%}',
                  f'{delta:+.1%} vs actual', dark=True), unsafe_allow_html=True)
r[2].markdown(kpi('Risk band', D.risk_band(new_p),
                  f'was {D.risk_band(base_p)} · red gauge marker is the actual value'),
              unsafe_allow_html=True)
with r[3]:
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=new_p*100,
        number={'suffix':'%','font':{'size':26,'color':NAVY}},
        gauge={'axis':{'range':[0,100],'tickwidth':1,'tickcolor':MUTED},
               'bar':{'color':NAVY,'thickness':.72},
               'bgcolor':'white','borderwidth':0,
               'steps':[{'range':[0,20],'color':'#e6f4e6'},
                        {'range':[20,45],'color':'#fdf3d9'},
                        {'range':[45,70],'color':'#fbe6dc'},
                        {'range':[70,100],'color':'#f7dcdc'}],
               'threshold':{'line':{'color':CRITICAL,'width':3},'thickness':.85,
                            'value':base_p*100}}))
    fig.update_layout(height=178, margin=dict(l=16,r=16,t=6,b=6))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ sensitivity
st.markdown('##### Which lever moves this supplier most')
st.caption('Each bar sweeps one variable across its plausible range while holding everything '
           'else at the values you set above.')
sens = []
for key, lab, lo, hi, step, scale in LEVERS:
    probs = []
    for v in np.linspace(lo, hi, 9):
        t = sim.copy(); t.loc[:, key] = v*scale
        probs.append(float(pipe.predict_proba(t)[:,1][0]))
    sens.append(dict(lever=lab, swing=max(probs)-min(probs),
                     best=min(probs), worst=max(probs)))
sens = pd.DataFrame(sens).sort_values('swing')
fig = go.Figure()
fig.add_bar(y=sens.lever, x=sens.swing*100, orientation='h', marker_color=BLUE,
            hovertemplate='<b>%{y}</b><br>Swings probability by %{x:.1f} points<extra></extra>')
fig.update_layout(xaxis_title='Probability swing across the variable range (percentage points)',
                  yaxis_title=None, showlegend=False)
st.plotly_chart(style(fig, height=380), use_container_width=True)

top_lever = sens.iloc[-1]
st.markdown(
    f'<div class="info"><b>For this supplier, {top_lever.lever.lower()} is the strongest single '
    f'lever</b>, swinging predicted failure probability by {top_lever.swing*100:.1f} percentage '
    f'points across its range. That is where a supplier development conversation should start. '
    f'Bear in mind these are associations learned from historical patterns, not proven causal '
    f'effects: improving a metric the model likes does not automatically reduce real risk unless '
    f'the underlying capability improves with it.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ trajectory
hist = D.scored()
hist = hist[hist.supplier_id == sel].sort_values('qi')
if len(hist) > 1:
    st.markdown('##### This supplier over time')
    fig = go.Figure()
    fig.add_scatter(x=hist.fiscal_quarter, y=hist.on_time_rate*100, name='On-time %',
                    mode='lines+markers', line=dict(color=BLUE, width=2), marker=dict(size=7),
                    hovertemplate='%{x}<br>On-time %{y:.1f}%<extra></extra>')
    fig.add_scatter(x=hist.fiscal_quarter, y=hist.external_credit_score, name='Credit score',
                    mode='lines+markers', line=dict(color=ORANGE, width=2, dash='dot'),
                    marker=dict(size=7),
                    hovertemplate='%{x}<br>Credit %{y:.0f}<extra></extra>')
    h2 = hist.dropna(subset=['prob_rf'])
    if not h2.empty:
        fig.add_scatter(x=h2.fiscal_quarter, y=h2.prob_rf*100, name='Predicted failure risk %',
                        mode='lines+markers', line=dict(color=CRITICAL, width=2.5),
                        marker=dict(size=8),
                        hovertemplate='%{x}<br>Risk %{y:.1f}%<extra></extra>')
    fig.update_layout(yaxis_title='Percent / score (all on one 0-100 scale)', xaxis_title=None)
    fig.update_xaxes(categoryorder='array', categoryarray=D.quarters(), tickangle=-45)
    st.plotly_chart(style(fig, height=330), use_container_width=True)
    st.caption('All three series share a 0-100 scale, so they can sit on one axis without '
               'a misleading second scale.')
