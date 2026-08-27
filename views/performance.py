import streamlit as st
import pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.express as px
from lib import data as D
from lib.theme import *

st.title('Delivery & Invoicing Performance')
st.markdown('<div class="sub">The descriptive layer: what actually happened, by supplier, '
            'category and quarter.</div>', unsafe_allow_html=True)
chips = D.active_filters()
if chips:
    st.markdown(''.join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)

sup = D.fs(); sq = D.fsq()
if sup.empty: st.warning('No suppliers match the current filters.'); st.stop()

tab1, tab2, tab3 = st.tabs(['Delivery reliability', 'Invoice accuracy', 'SLA compliance'])

# =========================================================== DELIVERY
with tab1:
    c = st.columns(4)
    ot = np.average(sq.on_time_rate.fillna(0), weights=sq.lines.fillna(0)) if sq.lines.sum() else 0
    inf= np.average(sq.in_full_rate.fillna(0), weights=sq.lines.fillna(0)) if sq.lines.sum() else 0
    c[0].markdown(kpi('On time', f'{ot*100:.1f}%', 'of delivered lines'), unsafe_allow_html=True)
    c[1].markdown(kpi('In full', f'{inf*100:.1f}%', 'of delivered lines'), unsafe_allow_html=True)
    c[2].markdown(kpi('Avg delay', f'{sq.avg_delay_days.mean():.2f} days',
                      'across all receipts'), unsafe_allow_html=True)
    c[3].markdown(kpi('Stockouts caused', f'{int(sq.stockouts.sum()):,}',
                      'downstream shortfalls'), unsafe_allow_html=True)
    st.write('')

    l, r = st.columns([1.3, 1])
    with l:
        st.markdown('##### On-time rate by category and quarter')
        piv = sq.pivot_table(index='primary_category', columns='fiscal_quarter',
                             values='on_time_rate', aggfunc='mean')
        cols = [q for q in D.quarters() if q in piv.columns]
        piv = piv[cols]*100
        if not piv.empty:
            fig = go.Figure(go.Heatmap(
                z=piv.values, x=piv.columns, y=piv.index,
                colorscale=[[0,'#cde2fb'],[0.5,'#3987e5'],[1,'#0d366b']],
                hovertemplate='<b>%{y}</b><br>%{x}<br>On-time %{z:.1f}%<extra></extra>',
                colorbar=dict(title=dict(text='%', side='right'), thickness=12, len=.85)))
            fig.update_layout(xaxis_title=None, yaxis_title=None)
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(style(fig, height=430), use_container_width=True)
            st.caption('Darker is better. Vertical bands of light colour mark quarters where '
                       'the whole network struggled, not individual supplier failures.')
    with r:
        st.markdown('##### Worst delivery performers')
        w = sup.nsmallest(12, 'otif_rate')[['supplier_name','otif_rate',
                                            'avg_delay_days','total_spend_inr']]
        w = w.assign(otif_rate=(w.otif_rate*100).round(1),
                     total_spend_inr=(w.total_spend_inr/1e7).round(1))
        w = w.rename(columns={'supplier_name':'Supplier','otif_rate':'OTIF %',
                              'avg_delay_days':'Delay','total_spend_inr':'₹cr'})
        st.dataframe(w, hide_index=True, height=430, column_config={
            'Supplier': st.column_config.TextColumn(width='medium'),
            'OTIF %': st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                      format='%.1f', width='small'),
            'Delay': st.column_config.NumberColumn(format='%.1f', width='small'),
            '₹cr': st.column_config.NumberColumn(format='%.1f', width='small')})

    st.markdown('##### Delay distribution by supplier tier')
    fig = go.Figure()
    for i, t in enumerate(['Strategic','Preferred','Transactional']):
        g = sq[sq.supplier_tier == t]
        if g.empty: continue
        fig.add_box(y=g.avg_delay_days, name=t, marker_color=SERIES[i], boxpoints=False,
                    line=dict(width=1.6))
    fig.update_layout(yaxis_title='Delay (days)', showlegend=False,
                      margin=dict(l=60))
    st.plotly_chart(style(fig, height=290), use_container_width=True)

# =========================================================== INVOICE
with tab2:
    ie = D.invoice_errors()
    ids = D.supplier_ids()
    ie = ie[ie.supplier_id.isin(ids)]
    c = st.columns(4)
    c[0].markdown(kpi('Invoice accuracy', f'{sup.invoice_accuracy.mean()*100:.1f}%',
                      'clean on first pass'), unsafe_allow_html=True)
    c[1].markdown(kpi('Dispute rate', f'{sup.dispute_rate.mean()*100:.1f}%',
                      'of invoices formally disputed'), unsafe_allow_html=True)
    c[2].markdown(kpi('Days to payment', f'{sup.avg_days_to_payment.mean():.0f}',
                      'average across suppliers'), unsafe_allow_html=True)
    c[3].markdown(kpi('Billing variance', inr(sup.abs_variance_inr.sum()),
                      'absolute, invoice vs PO'), unsafe_allow_html=True)
    st.write('')

    l, r = st.columns([1, 1.25])
    with l:
        st.markdown('##### What goes wrong on invoices')
        e = ie[ie.invoice_error_type != 'None'].groupby(
            'invoice_error_type', as_index=False)['size'].sum().sort_values('size')
        if not e.empty:
            fig = go.Figure(go.Bar(y=e.invoice_error_type, x=e['size'], orientation='h',
                marker_color=BLUE, text=e['size'], textposition='outside',
                textfont=dict(size=11, color=INK),
                hovertemplate='<b>%{y}</b><br>%{x:,} invoices<extra></extra>'))
            fig.update_layout(xaxis_title='Invoices affected', yaxis_title=None,
                              xaxis_range=[0, e['size'].max()*1.2])
            st.plotly_chart(style(fig, height=340), use_container_width=True)
    with r:
        st.markdown('##### Invoice accuracy against delivery reliability')
        fig = px.scatter(sup, x='otif_rate', y='invoice_accuracy', color='supplier_tier',
                         hover_name='supplier_name',
                         color_discrete_map={'Strategic':BLUE,'Preferred':ORANGE,
                                             'Transactional':'#c3c2b7'})
        fig.update_traces(marker=dict(size=8, opacity=.75, line=dict(color='white', width=.8)),
            hovertemplate='<b>%{hovertext}</b><br>OTIF %{x:.1%}<br>'
                          'Invoice accuracy %{y:.1%}<extra></extra>')
        fig.update_layout(xaxis_title='OTIF rate', yaxis_title='Invoice accuracy',
                          legend_title=None)
        fig.update_xaxes(tickformat='.0%'); fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(style(fig, height=340), use_container_width=True)
        corr = sup[['otif_rate','invoice_accuracy']].corr().iloc[0,1]
        st.caption(f'Correlation {corr:.2f}. Billing discipline and delivery discipline are '
                   f'{"related but distinct" if abs(corr)<0.6 else "closely linked"}, which is '
                   f'why the scorecard scores them separately.')

# =========================================================== SLA
with tab3:
    breach = sup[sup.sla_gap_pp < 0]
    c = st.columns(4)
    c[0].markdown(kpi('Below SLA', f'{len(breach)}', f'of {len(sup)} suppliers'),
                  unsafe_allow_html=True)
    c[1].markdown(kpi('Median gap', f'{sup.sla_gap_pp.median():+.1f} pp',
                      'actual minus contracted'), unsafe_allow_html=True)
    c[2].markdown(kpi('Worst gap', f'{sup.sla_gap_pp.min():+.1f} pp',
                      'single supplier'), unsafe_allow_html=True)
    c[3].markdown(kpi('Spend in breach', inr(breach.total_spend_inr.sum()),
                      'held by SLA-breaching suppliers', dark=True), unsafe_allow_html=True)
    st.write('')

    st.markdown('##### Contracted service level against delivered performance')
    d = sup.dropna(subset=['sla_target_pct']).copy()
    d['status'] = np.where(d.sla_gap_pp >= 0, 'Meeting SLA', 'In breach')
    fig = px.scatter(d, x='sla_target_pct', y=d.on_time_rate*100, color='status',
                     hover_name='supplier_name', size='total_spend_inr', size_max=26,
                     color_discrete_map={'Meeting SLA':GOOD, 'In breach':CRITICAL})
    lo = min(d.sla_target_pct.min(), (d.on_time_rate*100).min()) - 3
    hi = max(d.sla_target_pct.max(), (d.on_time_rate*100).max()) + 3
    fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(color=MUTED, dash='dash', width=1.5))
    fig.add_annotation(x=hi, y=hi, text='parity', showarrow=False,
                       xanchor='right', yshift=12, font=dict(size=10, color=MUTED))
    fig.update_traces(marker=dict(opacity=.75, line=dict(color='white', width=.8)),
        hovertemplate='<b>%{hovertext}</b><br>SLA target %{x:.0f}%<br>'
                      'Actual %{y:.1f}%<extra></extra>')
    fig.update_layout(xaxis_title='Contracted on-time target (%)',
                      yaxis_title='Actual on-time delivery (%)', legend_title=None)
    st.plotly_chart(style(fig, height=420), use_container_width=True)
    st.markdown('<div class="note"><b>Governance finding.</b> Points below the dashed parity '
                'line are suppliers delivering worse than the service level they signed. '
                'Penalty clauses average around 1% of order value across this base, which is '
                'evidently too small to change behaviour.</div>', unsafe_allow_html=True)
