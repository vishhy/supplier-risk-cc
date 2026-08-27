import streamlit as st
import pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.express as px
from lib import data as D
from lib.theme import *

st.title('Executive Overview')
st.markdown('<div class="sub">Where the money goes, how reliably it is delivered, '
            'and what is at risk next quarter.</div>', unsafe_allow_html=True)

chips = D.active_filters()
if chips:
    st.markdown(''.join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

sup = D.fs(); sq = D.fsq(); sc = D.fscored()
if sup.empty:
    st.warning('No suppliers match the current filters.'); st.stop()

# ------------------------------------------------------------------ KPI row
spend = sq.spend_inr.sum()
otif  = np.average(sq.otif_rate.fillna(0), weights=sq.lines.fillna(0)) if sq.lines.sum() else 0
ontime= np.average(sq.on_time_rate.fillna(0), weights=sq.lines.fillna(0)) if sq.lines.sum() else 0
breach= int((sup.sla_gap_pp < 0).sum())
latest = D.latest_scored_quarter()
cur = sc[sc.fiscal_quarter == latest]
at_risk = cur[cur.prob_rf >= 0.45]
risk_spend = at_risk.total_spend_inr.sum()

c = st.columns(5)
c[0].markdown(kpi('Spend in scope', inr(spend),
                  f'{len(sup)} suppliers · {sq.po_count.sum():,.0f} orders'), unsafe_allow_html=True)
c[1].markdown(kpi('On-time delivery', f'{ontime*100:.1f}%',
                  f'OTIF {otif*100:.1f}%'), unsafe_allow_html=True)
c[2].markdown(kpi('Below contracted SLA', f'{breach}',
                  f'of {len(sup)} suppliers'), unsafe_allow_html=True)
c[3].markdown(kpi('High-risk suppliers', f'{len(at_risk)}',
                  f'p ≥ 0.45, scored on {latest}'), unsafe_allow_html=True)
c[4].markdown(kpi('Spend at risk', inr(risk_spend),
                  'held by high-risk suppliers', dark=True), unsafe_allow_html=True)

st.write('')

# ------------------------------------------------------------------ row 1
left, right = st.columns([1.35, 1])

with left:
    st.markdown('##### Spend concentration')
    d = sup.sort_values('total_spend_inr', ascending=False).reset_index(drop=True)
    d['share'] = d.total_spend_inr/d.total_spend_inr.sum()*100
    d['cum'] = d.share.cumsum()
    d['rank_in_view'] = np.arange(1, len(d)+1)
    colour = {'A':BLUE, 'B':'#9ec5f4', 'C':'#d8d7d0'}
    fig = go.Figure()
    for cls in ['A','B','C']:
        g = d[d.abc_class == cls]
        if g.empty: continue
        fig.add_bar(x=g.rank_in_view, y=g.share, name=f'Class {cls}',
                    marker_color=colour[cls], marker_line_width=0,
                    customdata=np.stack([g.supplier_name, g.total_spend_inr/1e7, g.cum], axis=-1),
                    hovertemplate='<b>%{customdata[0]}</b><br>Rank %{x}<br>'
                                  'Share %{y:.2f}%<br>Spend ₹%{customdata[1]:,.1f} cr<br>'
                                  'Cumulative %{customdata[2]:.1f}%<extra></extra>')
    fig.update_layout(barmode='stack', bargap=0, legend_traceorder='normal',
                      xaxis_title='Suppliers ranked by spend', yaxis_title='Share of spend (%)')
    st.plotly_chart(style(fig, height=330), use_container_width=True)
    top20 = d.head(20).share.sum()
    st.caption(f'Top 20 suppliers hold {top20:.1f}% of spend in the current selection. '
               f'Class A: {int((d.abc_class=="A").sum())} suppliers.')

with right:
    st.markdown('##### Delivery reliability over time')
    t = sq.groupby('fiscal_quarter').apply(
        lambda g: pd.Series({
            'on_time': np.average(g.on_time_rate.fillna(0), weights=g.lines.fillna(0))*100
                       if g.lines.sum() else np.nan,
            'lines': g.lines.sum()})).reset_index()
    order = [q for q in D.quarters() if q in set(t.fiscal_quarter)]
    t['o'] = t.fiscal_quarter.map({q:i for i,q in enumerate(D.quarters())})
    t = t.sort_values('o')
    fig = go.Figure()
    for i, q in enumerate(order):
        if q.endswith('Q3'):
            fig.add_vrect(x0=i-0.45, x1=i+0.45, fillcolor=WARNING, opacity=.18,
                          line_width=0, layer='below')
    fig.add_scatter(x=t.fiscal_quarter, y=t.on_time, mode='lines+markers',
                    line=dict(color=BLUE, width=2.5), marker=dict(size=8),
                    name='On-time %',
                    hovertemplate='<b>%{x}</b><br>On-time %{y:.1f}%<extra></extra>')
    fig.update_layout(yaxis_title='On-time delivery (%)', xaxis_title=None, showlegend=False)
    fig.update_xaxes(categoryorder='array', categoryarray=order, tickangle=-45)
    st.plotly_chart(style(fig, height=330), use_container_width=True)
    q3 = t[t.fiscal_quarter.str.endswith('Q3')].on_time.mean()
    oth = t[~t.fiscal_quarter.str.endswith('Q3')].on_time.mean()
    if not np.isnan(q3) and not np.isnan(oth):
        st.caption(f'Shaded bands are festive Q3 quarters, averaging {q3:.1f}% on-time '
                   f'against {oth:.1f}% elsewhere.')

# ------------------------------------------------------------------ row 2
left, right = st.columns([1, 1])

with left:
    st.markdown('##### Category performance')
    cat = sq.groupby('primary_category').apply(lambda g: pd.Series({
        'spend': g.spend_inr.sum()/1e7,
        'on_time': np.average(g.on_time_rate.fillna(0), weights=g.lines.fillna(0))*100
                   if g.lines.sum() else np.nan,
        'suppliers': g.supplier_id.nunique()})).reset_index().dropna()
    if not cat.empty:
        fig = px.scatter(cat, x='on_time', y='spend', size='suppliers',
                         hover_name='primary_category', size_max=42)
        fig.update_traces(marker=dict(color=BLUE, opacity=.8,
                                      line=dict(color='white', width=1.5)),
            hovertemplate='<b>%{hovertext}</b><br>On-time %{x:.1f}%<br>'
                          'Spend ₹%{y:,.0f} cr<extra></extra>')
        med = cat.on_time.median()
        fig.add_vline(x=med, line_dash='dash', line_color=MUTED, line_width=1)
        # label the three weakest and the two largest, with leader lines so text never
        # sits on top of a mark
        notable = pd.concat([cat.nsmallest(3,'on_time'), cat.nlargest(2,'spend')]
                            ).drop_duplicates('primary_category')
        for i, (_, r) in enumerate(notable.iterrows()):
            fig.add_annotation(x=r.on_time, y=r.spend, text=r.primary_category,
                               showarrow=True, arrowhead=0, arrowwidth=1,
                               arrowcolor=MUTED, ax=0, ay=-34 if i % 2 == 0 else 34,
                               font=dict(size=10, color=INK2))
        fig.update_layout(xaxis_title='On-time delivery (%)', yaxis_title='Spend (₹ crore)')
        st.plotly_chart(style(fig, height=380), use_container_width=True)
        st.caption('Bubble size is the number of suppliers. Dashed line is the median '
                   'category. Hover any bubble for its name; the weakest three and largest '
                   'two are labelled.')

with right:
    st.markdown('##### Risk profile, next quarter')
    if cur.empty:
        st.info('The selected quarter range contains no scored quarter. '
                'Extend it to include ' + D.scored_quarters()[0] + ' or later.')
    else:
        b = cur.copy()
        b['band'] = b.prob_rf.apply(D.risk_band)
        agg = b.groupby('band').agg(suppliers=('supplier_id','count'),
                                    spend=('total_spend_inr','sum')).reset_index()
        order_b = ['Critical','High','Moderate','Low']
        agg['o'] = agg.band.map({k:i for i,k in enumerate(order_b)})
        agg = agg.sort_values('o')
        fig = go.Figure()
        fig.add_bar(y=agg.band, x=agg.suppliers, orientation='h',
                    marker_color=[RISK_COLOURS[b] for b in agg.band],
                    text=[f'{n}  ·  {inr(s)}' for n, s in zip(agg.suppliers, agg.spend)],
                    textposition='outside', textfont=dict(size=11, color=INK),
                    hovertemplate='<b>%{y} risk</b><br>%{x} suppliers<extra></extra>')
        fig.update_layout(xaxis_title='Suppliers', yaxis_title=None, showlegend=False,
                          xaxis_range=[0, agg.suppliers.max()*1.45])
        fig.update_yaxes(categoryorder='array', categoryarray=order_b[::-1])
        st.plotly_chart(style(fig, height=330), use_container_width=True)
        crit = agg[agg.band.isin(['Critical','High'])]
        st.caption(f'Scored on {latest}, predicting the quarter that follows it. '
                   f'{int(crit.suppliers.sum())} suppliers carrying {inr(crit.spend.sum())} '
                   f'sit in the high or critical bands.')

st.markdown(
    '<div class="note"><b>Read this alongside the honest benchmark.</b> The model reaches '
    '0.890 out-of-sample AUC, but ranking suppliers on their two-quarter rolling on-time rate '
    'alone reaches 0.878. Most of the value on this page comes from ranking systematically at '
    'all, not from the algorithm. The Risk Prediction page quantifies that gap.</div>',
    unsafe_allow_html=True)
