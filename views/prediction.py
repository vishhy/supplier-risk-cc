import streamlit as st
import pandas as pd, numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
from lib import data as D
from lib.theme import *

st.title('Risk Prediction')
st.markdown('<div class="sub">Out-of-sample predictions of severe delivery failure in the '
            'following quarter, and the economics of acting on them.</div>',
            unsafe_allow_html=True)
chips = D.active_filters()
if chips:
    st.markdown(''.join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

sc = D.fscored()
mm = D.model_metrics()
T = 'target_delivery_failure_next_q'
if sc.empty:
    st.warning('No scored supplier-quarters in this selection. Scored quarters are '
               + ', '.join(D.scored_quarters()) + '.')
    st.stop()

MODELS = {'Random Forest':'prob_rf', 'Logistic Regression':'prob_lr',
          'Single column (2-qtr rolling on-time rate)':'prob_benchmark'}

top = st.columns([1.6, 1, 1])
choice = top[0].radio('Model', list(MODELS), horizontal=True, key='model_choice')
col = MODELS[choice]
cost_miss   = top[1].number_input('Cost of a missed failure (₹ lakh)', 1.0, 50.0,
                                  8.5, 0.5, key='cm') * 1e5
cost_action = top[2].number_input('Cost of a mitigation action (₹ thousand)', 5, 500,
                                  60, 5, key='ca') * 1e3

y = sc[T].values
p = sc[col].values
ok = ~np.isnan(p)
y, p, sub = y[ok], p[ok], sc[ok]
auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

# ------------------------------------------------------------------ threshold
st.markdown('##### Intervention threshold')
tbl = D.cost_table(p, y, cost_miss, cost_action)
opt = tbl.loc[tbl.cost.idxmin()]
thr = st.slider('Flag a supplier when predicted probability is at least',
                0.02, 0.98, float(opt.threshold), 0.02, key='thr')

pred = p >= thr
tp = int((pred & (y==1)).sum()); fp = int((pred & (y==0)).sum())
fn = int((~pred & (y==1)).sum()); tn = int((~pred & (y==0)).sum())
cost = fn*cost_miss + (tp+fp)*cost_action
donothing = int(y.sum())*cost_miss
prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)

c = st.columns(5)
c[0].markdown(kpi('AUC', f'{auc:.3f}' if not np.isnan(auc) else 'n/a',
                  f'{len(y):,} supplier-quarters'), unsafe_allow_html=True)
c[1].markdown(kpi('Suppliers flagged', f'{tp+fp:,}', f'{(tp+fp)/len(y)*100:.0f}% of the base'),
              unsafe_allow_html=True)
c[2].markdown(kpi('Precision', f'{prec*100:.1f}%', 'of flagged that do fail'),
              unsafe_allow_html=True)
c[3].markdown(kpi('Recall', f'{rec*100:.1f}%', 'of failures caught'), unsafe_allow_html=True)
c[4].markdown(kpi('Expected cost', inr(cost),
                  f'vs {inr(donothing)} doing nothing', dark=True), unsafe_allow_html=True)

if abs(thr - opt.threshold) > 0.001:
    delta = cost - opt.cost
    st.caption(f'Cost-minimising threshold for these assumptions is {opt.threshold:.2f} '
               f'({inr(opt.cost)}). You are {inr(abs(delta))} '
               f'{"above" if delta>0 else "below"} that.')

st.write('')
l, m, r = st.columns([1.15, 1, 1])

with l:
    st.markdown('##### Cost curve')
    fig = go.Figure()
    lo = tbl[tbl.cost <= tbl.cost.min()*1.05]
    if len(lo) > 1:
        fig.add_vrect(x0=lo.threshold.min(), x1=lo.threshold.max(),
                      fillcolor=WARNING, opacity=.16, line_width=0, layer='below')
    fig.add_scatter(x=tbl.threshold, y=tbl.cost/1e7, mode='lines',
                    line=dict(color=BLUE, width=2.5), name='Expected cost',
                    hovertemplate='Threshold %{x:.2f}<br>₹%{y:,.2f} cr<extra></extra>')
    fig.add_scatter(x=[thr], y=[cost/1e7], mode='markers', name='Your setting',
                    marker=dict(color=CRITICAL, size=13, line=dict(color='white', width=2)),
                    hovertemplate='Your setting %{x:.2f}<br>₹%{y:,.2f} cr<extra></extra>')
    fig.update_layout(xaxis_title='Threshold', yaxis_title='Expected cost (₹ crore)',
                      showlegend=False)
    st.plotly_chart(style(fig, height=310), use_container_width=True)
    if len(lo) > 1:
        st.caption(f'Shaded band is within 5% of the minimum: {lo.threshold.min():.2f} to '
                   f'{lo.threshold.max():.2f}. The optimum is a plateau, not a precise point.')

with m:
    st.markdown('##### Outcomes at this threshold')
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(z=z, x=['Not flagged','Flagged'], y=['No failure','Failure'],
        colorscale=[[0,'#eef3fb'],[1,'#2a78d6']], showscale=False,
        text=[[f'{tn:,}<br>correct pass', f'{fp:,}<br>false alarm'],
              [f'{fn:,}<br>missed failure', f'{tp:,}<br>caught']],
        texttemplate='%{text}', textfont=dict(size=12),
        hovertemplate='%{y} / %{x}: %{z:,}<extra></extra>'))
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(style(fig, height=310), use_container_width=True)
    st.caption(f'Each missed failure is assumed to cost {inr(cost_miss)}; each action '
               f'{inr(cost_action)}.')

with r:
    st.markdown('##### Model against the one-column rule')
    fig = go.Figure()
    for name, c_, colour in [('Random Forest','prob_rf',BLUE),
                             ('Logistic Regression','prob_lr',ORANGE),
                             ('Single column','prob_benchmark',MUTED)]:
        v = sc[c_].values; k = ~np.isnan(v)
        if len(np.unique(sc[T].values[k])) < 2: continue
        fpr, tpr, _ = roc_curve(sc[T].values[k], v[k])
        a = roc_auc_score(sc[T].values[k], v[k])
        fig.add_scatter(x=fpr, y=tpr, mode='lines', name=f'{name} · {a:.3f}',
                        line=dict(color=colour, width=2.2 if c_!='prob_benchmark' else 1.8,
                                  dash='dash' if c_=='prob_benchmark' else 'solid'),
                        hovertemplate=f'{name}<br>FPR %{{x:.2f}} · TPR %{{y:.2f}}<extra></extra>')
    fig.add_scatter(x=[0,1], y=[0,1], mode='lines', showlegend=False,
                    line=dict(color=GRID, dash='dot', width=1.5), hoverinfo='skip')
    fig.update_layout(xaxis_title='False positive rate', yaxis_title='True positive rate')
    st.plotly_chart(style(fig, height=310), use_container_width=True)

# ------------------------------------------------------------------ capacity
st.markdown('##### Capacity-constrained triage')
st.markdown('<div class="sub">The cost optimum above tends to flag a very large share of the '
            'base, because a missed failure costs far more than a wasted action. In practice '
            'the binding constraint is how many supplier reviews the team can actually run.'
            '</div>', unsafe_allow_html=True)

cap = st.slider('Supplier reviews the team can run per quarter', 5, 120, 40, 5, key='cap')
nq = sub.fiscal_quarter.nunique()
rows = []
for name, c_ in [('Random Forest','prob_rf'), ('Logistic Regression','prob_lr'),
                 ('Single column rule','prob_benchmark')]:
    caught = 0; reviewed = 0
    for q, g in sub.groupby('fiscal_quarter'):
        gg = g.dropna(subset=[c_]).nlargest(min(cap, len(g)), c_)
        caught += int(gg[T].sum()); reviewed += len(gg)
    total = int(sub[T].sum())
    rows.append(dict(Model=name, Reviewed=reviewed, Caught=caught,
                     Precision=caught/max(reviewed,1), Recall=caught/max(total,1)))
cap_df = pd.DataFrame(rows)

cc = st.columns([1.3, 1])
with cc[0]:
    fig = go.Figure()
    fig.add_bar(x=cap_df.Model, y=cap_df.Recall*100, name='Failures caught (%)',
                marker_color=[BLUE, ORANGE, MUTED],
                text=[f'{v*100:.0f}%' for v in cap_df.Recall], textposition='outside',
                textfont=dict(size=12, color=INK),
                hovertemplate='<b>%{x}</b><br>%{y:.1f}% of failures caught<extra></extra>')
    fig.update_layout(yaxis_title='Failures caught (%)', xaxis_title=None, showlegend=False,
                      yaxis_range=[0, min(100, cap_df.Recall.max()*130)])
    st.plotly_chart(style(fig, height=290), use_container_width=True)
with cc[1]:
    cap_show = cap_df.assign(Precision=(cap_df.Precision*100).round(1),
                             Recall=(cap_df.Recall*100).round(1)).rename(
        columns={'Precision':'Precision %','Recall':'Recall %'})
    st.dataframe(cap_show, hide_index=True, height=290, column_config={
        'Precision %': st.column_config.NumberColumn(format='%.1f'),
        'Recall %': st.column_config.NumberColumn(format='%.1f')})
    rf_r = cap_df.loc[cap_df.Model=='Random Forest','Caught'].iloc[0]
    bn_r = cap_df.loc[cap_df.Model=='Single column rule','Caught'].iloc[0]
    extra = rf_r - bn_r
    st.caption(f'At {cap} reviews a quarter across {nq} quarters, the model catches '
               f'{extra:+d} more failures than the one-column rule, worth '
               f'{inr(abs(extra)*cost_miss)}.')

# ------------------------------------------------------------------ watchlist
st.markdown('##### Action watchlist')
latest = sub.fiscal_quarter.map({q:i for i,q in enumerate(D.quarters())}).idxmax()
lq = sub.loc[latest,'fiscal_quarter']
w = sub[sub.fiscal_quarter == lq].copy()
w['exposure'] = w.spend_share_of_quarter_pct*(1+w.is_sole_source)/(1+w.num_alternate_suppliers)
w['priority'] = w[col]*(1+w.exposure)
w['band'] = w[col].apply(D.risk_band)
w = w.sort_values('priority', ascending=False)
flagged = w[w[col] >= thr]

st.caption(f'Scoring quarter {lq}: {len(flagged)} of {len(w)} suppliers clear the '
           f'{thr:.2f} threshold, holding {flagged.spend_share_of_quarter_pct.sum():.1f}% '
           f'of that quarter\'s spend.')
flagged = flagged.assign(on_time_rate=(flagged.on_time_rate*100).round(1),
                         total_spend_inr=(flagged.total_spend_inr/1e7).round(2))
show = flagged[['supplier_name','supplier_tier','primary_category',col,'band',
                'spend_share_of_quarter_pct','is_sole_source','num_alternate_suppliers',
                'on_time_rate','external_credit_score','total_spend_inr']].rename(columns={
    'supplier_name':'Supplier','supplier_tier':'Tier','primary_category':'Category',
    col:'Failure probability','band':'Risk band',
    'spend_share_of_quarter_pct':'Spend share %','is_sole_source':'Sole source',
    'num_alternate_suppliers':'Alternatives','on_time_rate':'On-time %',
    'external_credit_score':'Credit','total_spend_inr':'Spend ₹cr'})
st.dataframe(show, hide_index=True, height=380, column_config={
    'Failure probability': st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                           format='%.2f'),
    'Spend share %': st.column_config.NumberColumn(format='%.2f'),
    'Sole source': st.column_config.CheckboxColumn(),
    'On-time %': st.column_config.NumberColumn(format='%.1f'),
    'Credit': st.column_config.NumberColumn(format='%.0f'),
    'Spend ₹cr': st.column_config.NumberColumn(format='%.2f')})
st.download_button('Download watchlist as CSV', show.to_csv(index=False).encode(),
                   f'watchlist_{lq}.csv', 'text/csv')

# ------------------------------------------------------------------ drivers
with st.expander('What the model is keying on'):
    fi = D.feature_importance().head(15).sort_values('importance')
    FIN = {'external_credit_score','altman_z_score','days_sales_outstanding',
           'debt_to_equity_ratio','current_ratio','yoy_revenue_growth_pct',
           'news_sentiment_score','gst_filing_delay_flag','litigation_case_count',
           'promoter_pledge_pct','credit_score_change','event_count','critical_event_count'}
    fig = go.Figure(go.Bar(x=fi.importance, y=fi.feature.str.replace('_',' '), orientation='h',
        marker_color=[ORANGE if f in FIN else BLUE for f in fi.feature],
        hovertemplate='<b>%{y}</b><br>%{x:.3f}<extra></extra>'))
    fig.update_layout(xaxis_title='Random Forest importance', yaxis_title=None)
    st.plotly_chart(style(fig, height=420), use_container_width=True)
    st.markdown('Blue is operational and commercial, orange is external financial. '
                'They interleave rather than separating, because both are noisy readings of '
                'the same underlying supplier condition.')

st.markdown(
    f'<div class="note"><b>The honest comparison.</b> On the full out-of-sample set the '
    f'Random Forest reaches {mm["auc_rf"]:.3f} AUC and the single-column rule reaches '
    f'{mm["auc_benchmark"]:.3f}. That gap is not statistically significant across folds '
    f'(p = 0.09). Switch the model selector to the single column above and watch how little '
    f'changes. The model earns its place under a capacity constraint, and only narrowly.</div>',
    unsafe_allow_html=True)
