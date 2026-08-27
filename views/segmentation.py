import streamlit as st
import pandas as pd, numpy as np
import plotly.graph_objects as go
from lib import data as D
from lib.theme import *

st.title('Portfolio Strategy')
st.markdown('<div class="sub">Kraljic positioning: what to do about a supplier depends on '
            'exposure as much as on probability.</div>', unsafe_allow_html=True)
chips = D.active_filters()
if chips:
    st.markdown(''.join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

sc = D.fscored()
if sc.empty:
    st.warning('No scored supplier-quarters in this selection.'); st.stop()

lq = D.latest_scored_quarter()
if lq not in set(sc.fiscal_quarter):
    lq = sc.sort_values('qi').fiscal_quarter.iloc[-1]
seg = sc[sc.fiscal_quarter == lq].copy()

c = st.columns([1.2,1,1])
model_col = c[0].radio('Risk source', ['Random Forest','Single column rule'],
                       horizontal=True, key='seg_model')
pcol = 'prob_rf' if model_col == 'Random Forest' else 'prob_benchmark'
risk_cut  = c[1].slider('Supply risk cut-off (percentile)', 50, 90, 70, 5, key='seg_rcut')
imp_cut   = c[2].slider('Profit impact cut-off (percentile)', 50, 90, 70, 5, key='seg_icut')

seg['supply_risk'] = (seg[pcol]*0.6 + seg.is_sole_source*0.20
                      + (1/(1+seg.num_alternate_suppliers))*0.20)
seg['profit_impact'] = seg.spend_share_of_quarter_pct * seg.category_criticality.fillna(0.5)
rq = seg.supply_risk.quantile(risk_cut/100); iq = seg.profit_impact.quantile(imp_cut/100)
seg['quadrant'] = np.select(
    [(seg.supply_risk>=rq)&(seg.profit_impact>=iq),
     (seg.supply_risk>=rq)&(seg.profit_impact< iq),
     (seg.supply_risk< rq)&(seg.profit_impact>=iq)],
    ['Strategic','Bottleneck','Leverage'], default='Non-critical')

ACTIONS = {
 'Strategic':   'Dual-source now, executive review, joint continuity plan',
 'Bottleneck':  'Qualify an alternate, hold buffer stock, redesign out where possible',
 'Leverage':    'Competitive tendering, hold performance, protect the margin',
 'Non-critical':'Automate ordering, consolidate vendors, review annually'}

T = 'target_delivery_failure_next_q'
summary = seg.groupby('quadrant').agg(
    Suppliers=('supplier_id','count'),
    Spend=('total_spend_inr','sum'),
    SpendShare=('spend_share_of_quarter_pct','sum'),
    MeanRisk=(pcol,'mean'),
    ActualFailure=(T,'mean')).reindex(
    ['Strategic','Bottleneck','Leverage','Non-critical']).fillna(0).reset_index()
summary['Action'] = summary.quadrant.map(ACTIONS)

k = st.columns(4)
for i, q in enumerate(['Strategic','Bottleneck','Leverage','Non-critical']):
    r = summary[summary.quadrant == q].iloc[0]
    k[i].markdown(kpi(q, f'{int(r.Suppliers)} suppliers',
                      f'{r.SpendShare:.1f}% of spend · {r.ActualFailure*100:.0f}% failed'),
                  unsafe_allow_html=True)
st.write('')

l, r = st.columns([1.35, 1])
with l:
    st.markdown(f'##### Portfolio position, {lq}')
    fig = go.Figure()
    for q in ['Non-critical','Leverage','Bottleneck','Strategic']:
        g = seg[seg.quadrant == q]
        if g.empty: continue
        fig.add_scatter(x=g.supply_risk, y=g.profit_impact, mode='markers',
            name=f'{q} ({len(g)})',
            marker=dict(color=KRALJIC_COLOURS[q], size=10, opacity=.8,
                        line=dict(color='white', width=1)),
            customdata=np.stack([g.supplier_name, g[pcol], g.spend_share_of_quarter_pct,
                                 g.is_sole_source], axis=-1),
            hovertemplate='<b>%{customdata[0]}</b><br>Failure probability %{customdata[1]:.2f}'
                          '<br>Spend share %{customdata[2]:.2f}%'
                          '<br>Sole source: %{customdata[3]}<extra></extra>')
    fig.add_vline(x=rq, line_dash='dash', line_color=MUTED, line_width=1.2)
    fig.add_hline(y=iq, line_dash='dash', line_color=MUTED, line_width=1.2)
    fig.update_layout(xaxis_title='Supply risk (probability, sole-sourcing, alternatives)',
                      yaxis_title='Profit impact (spend share × criticality)')
    # decade ticks only: minor log ticks print bare mantissas and repeat "2" and "5"
    # at every decade, which is unreadable
    fig.update_yaxes(type='log', dtick=1, tickformat='.3~g', showexponent='none',
                     minor=dict(showgrid=False))
    st.plotly_chart(style(fig, height=470), use_container_width=True)
    st.caption('Hover any point for the supplier. Move the cut-off sliders to see how '
               'sensitive the quadrant boundaries are.')

with r:
    st.markdown('##### Where the spend sits against where the risk sits')
    fig = go.Figure()
    fig.add_bar(x=summary.quadrant, y=summary.SpendShare, name='Share of spend (%)',
                marker_color=[KRALJIC_COLOURS[q] for q in summary.quadrant],
                text=[f'{v:.1f}%' for v in summary.SpendShare], textposition='outside',
                textfont=dict(size=11, color=INK),
                hovertemplate='<b>%{x}</b><br>%{y:.1f}% of spend<extra></extra>')
    fig.update_layout(yaxis_title='Share of quarter spend (%)', xaxis_title=None,
                      showlegend=False,
                      yaxis_range=[0, max(summary.SpendShare.max()*1.3, 1)])
    st.plotly_chart(style(fig, height=225), use_container_width=True)

    fig2 = go.Figure()
    fig2.add_bar(x=summary.quadrant, y=summary.ActualFailure*100,
                 marker_color=[KRALJIC_COLOURS[q] for q in summary.quadrant],
                 text=[f'{v*100:.0f}%' for v in summary.ActualFailure], textposition='outside',
                 textfont=dict(size=11, color=INK),
                 hovertemplate='<b>%{x}</b><br>%{y:.0f}% actually failed<extra></extra>')
    fig2.update_layout(yaxis_title='Actually failed next quarter (%)', xaxis_title=None,
                       showlegend=False,
                       yaxis_range=[0, max(summary.ActualFailure.max()*130, 10)])
    st.plotly_chart(style(fig2, height=225), use_container_width=True)

st.markdown('##### Recommended posture by quadrant')
disp = summary.assign(ActualFailure=(summary.ActualFailure*100).round(0))[
    ['quadrant','Suppliers','SpendShare','MeanRisk','ActualFailure','Action']].rename(
    columns={'quadrant':'Quadrant','SpendShare':'Spend share %',
             'MeanRisk':'Mean predicted risk','ActualFailure':'Actually failed %'})
st.dataframe(disp, hide_index=True, column_config={
    'Spend share %': st.column_config.NumberColumn(format='%.1f'),
    'Mean predicted risk': st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                           format='%.2f'),
    'Actually failed %': st.column_config.NumberColumn(format='%.0f'),
    'Action': st.column_config.TextColumn(width='large')})

st.markdown('##### Suppliers by quadrant')
pickq = st.selectbox('Show', ['Strategic','Bottleneck','Leverage','Non-critical'], key='seg_pick')
g = seg[seg.quadrant == pickq].sort_values('supply_risk', ascending=False)
g = g.assign(on_time_rate=(g.on_time_rate*100).round(1))
tbl = g[['supplier_name','supplier_tier','primary_category','region',pcol,
         'spend_share_of_quarter_pct','is_sole_source','num_alternate_suppliers',
         'on_time_rate','external_credit_score']].rename(columns={
    'supplier_name':'Supplier','supplier_tier':'Tier','primary_category':'Category',
    'region':'Region',pcol:'Failure probability',
    'spend_share_of_quarter_pct':'Spend share %','is_sole_source':'Sole source',
    'num_alternate_suppliers':'Alternatives','on_time_rate':'On-time %',
    'external_credit_score':'Credit'})
st.dataframe(tbl, hide_index=True, height=340, column_config={
    'Failure probability': st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                           format='%.2f'),
    'Spend share %': st.column_config.NumberColumn(format='%.2f'),
    'Sole source': st.column_config.CheckboxColumn(),
    'On-time %': st.column_config.NumberColumn(format='%.1f'),
    'Credit': st.column_config.NumberColumn(format='%.0f')})
st.download_button(f'Download {pickq} list as CSV', tbl.to_csv(index=False).encode(),
                   f'{pickq.lower()}_suppliers.csv', 'text/csv')

st.markdown(
    '<div class="note"><b>One caveat worth stating.</b> The supply risk axis uses model output, '
    'so quadrant membership is not independent of the prediction. The observed failure rates in '
    'the table confirm the ranking works; they are not a separate validation of it.</div>',
    unsafe_allow_html=True)
