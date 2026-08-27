import streamlit as st
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lib import data as D
from lib.theme import *

st.title('Supplier 360')
st.markdown('<div class="sub">Everything the data holds on one supplier: delivery record, '
            'billing behaviour, financial trajectory and logged incidents.</div>',
            unsafe_allow_html=True)

sup = D.fs()
if sup.empty: st.warning('No suppliers match the current filters.'); st.stop()

order = sup.sort_values('total_spend_inr', ascending=False)
sel = st.selectbox('Supplier', options=list(order.supplier_id),
                   format_func=lambda i: order.loc[order.supplier_id==i,'supplier_name'].iloc[0],
                   key='s360')
s = sup[sup.supplier_id == sel].iloc[0]
sq = D.supplier_quarter()
sq = sq[sq.supplier_id == sel].sort_values('qi')
sc = D.scored(); sc = sc[sc.supplier_id == sel].sort_values('qi')
ev = D.events(); ev = ev[ev.supplier_id == sel]

# ------------------------------------------------------------------ identity
st.markdown(
    f"<div style='background:{NAVY};border-radius:12px;padding:18px 22px;margin-bottom:14px'>"
    f"<div style='color:#fff;font-size:1.35rem;font-weight:700'>{s.supplier_name}</div>"
    f"<div style='color:#A8B4C8;font-size:.87rem;margin-top:4px'>"
    f"{s.supplier_tier} · {s.primary_category} · {s.city}, {s.state} · "
    f"vendor {s.vendor_code} · onboarded {s.onboarding_date}</div></div>",
    unsafe_allow_html=True)

c = st.columns(6)
latest_p = sc.dropna(subset=['prob_rf'])
p = latest_p.prob_rf.iloc[-1] if not latest_p.empty else np.nan
c[0].markdown(kpi('Total spend', inr(s.total_spend_inr), f'{int(s.po_count):,} orders'),
              unsafe_allow_html=True)
c[1].markdown(kpi('OTIF', f'{s.otif_rate*100:.1f}%',
                  f'on-time {s.on_time_rate*100:.0f}%'), unsafe_allow_html=True)
c[2].markdown(kpi('Invoice accuracy', f'{s.invoice_accuracy*100:.1f}%',
                  f'disputes {s.dispute_rate*100:.0f}%'), unsafe_allow_html=True)
c[3].markdown(kpi('Credit score', f'{s.external_credit_score:.0f}',
                  f'rating {s.credit_rating}'), unsafe_allow_html=True)
c[4].markdown(kpi('SLA gap', f'{s.sla_gap_pp:+.1f} pp',
                  f'target {s.sla_target_pct:.0f}%'), unsafe_allow_html=True)
c[5].markdown(kpi('Failure risk', f'{p:.0%}' if pd.notna(p) else 'n/a',
                  D.risk_band(p) if pd.notna(p) else 'not scored', dark=True),
              unsafe_allow_html=True)

flags = []
if s.is_sole_source: flags.append('Sole-sourced, no qualified alternative')
if s.num_alternate_suppliers == 0 and not s.is_sole_source:
    flags.append('No alternates currently qualified')
if s.sla_gap_pp < 0: flags.append(f'Below contracted SLA by {abs(s.sla_gap_pp):.1f} points')
if s.promoter_pledge_pct and s.promoter_pledge_pct > 50:
    flags.append(f'Promoter pledge at {s.promoter_pledge_pct:.0f}%')
if s.altman_z_score is not None and s.altman_z_score < 1.8:
    flags.append(f'Altman Z at {s.altman_z_score:.2f}, inside the distress zone')
if s.critical_events > 0: flags.append(f'{int(s.critical_events)} high or critical incidents')
if flags:
    st.markdown('<div class="note"><b>Attention flags.</b> ' + ' · '.join(flags) + '</div>',
                unsafe_allow_html=True)
st.write('')

t1, t2, t3, t4 = st.tabs(['Delivery record', 'Financial trajectory', 'Incidents', 'Peer comparison'])

with t1:
    l, r = st.columns([1.4, 1])
    with l:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            row_heights=[0.68, 0.32],
                            subplot_titles=('Delivery performance by quarter',
                                            'Lines delivered'))
        fig.add_scatter(x=sq.fiscal_quarter, y=sq.on_time_rate*100, name='On-time %',
                        mode='lines+markers', line=dict(color=BLUE, width=2.5),
                        marker=dict(size=8), row=1, col=1,
                        hovertemplate='%{x}<br>On-time %{y:.1f}%<extra></extra>')
        fig.add_scatter(x=sq.fiscal_quarter, y=sq.otif_rate*100, name='OTIF %',
                        mode='lines+markers', line=dict(color=ORANGE, width=2, dash='dot'),
                        marker=dict(size=7), row=1, col=1,
                        hovertemplate='%{x}<br>OTIF %{y:.1f}%<extra></extra>')
        fig.add_bar(x=sq.fiscal_quarter, y=sq.lines, name='Lines delivered',
                    marker_color='#c7d9f2', row=2, col=1, showlegend=False,
                    hovertemplate='%{x}<br>%{y} lines<extra></extra>')
        fig.update_yaxes(title_text='Percent', range=[0,105], row=1, col=1)
        fig.update_yaxes(title_text='Lines', rangemode='tozero', row=2, col=1)
        fig.update_xaxes(categoryorder='array', categoryarray=D.quarters(),
                         tickangle=-45, row=2, col=1)
        fig.update_annotations(font=dict(size=13, color=NAVY))
        st.plotly_chart(style(fig, height=430), use_container_width=True)
        st.caption('Volume sits in its own panel rather than on a second y-axis, so the two '
                   'scales cannot be misread against each other.')
    with r:
        # a bar chart hides quarters where the average delay is exactly zero, which
        # reads as missing data; markers make the zero visible
        fig = go.Figure()
        fig.add_scatter(x=sq.fiscal_quarter, y=sq.avg_delay_days, mode='lines+markers',
                        line=dict(color=SERIOUS, width=2), marker=dict(size=9),
                        hovertemplate='%{x}<br>Avg delay %{y:.2f} days<extra></extra>')
        fig.add_hline(y=0, line_color='#c3c2b7', line_width=1)
        fig.update_layout(title='Average delay', yaxis_title='Days', xaxis_title=None,
                          showlegend=False)
        fig.update_xaxes(categoryorder='array', categoryarray=D.quarters(), tickangle=-45)
        st.plotly_chart(style(fig, height=360), use_container_width=True)

    st.markdown('##### Quarter detail')
    d = sq[['fiscal_quarter','lines','po_count','spend_inr','on_time_rate','in_full_rate',
            'otif_rate','avg_delay_days','stockouts','invoice_error_rate']].copy()
    for c_ in ['on_time_rate','in_full_rate','otif_rate','invoice_error_rate']:
        d[c_] = (d[c_]*100).round(1)
    d['spend_inr'] = (d.spend_inr/1e7).round(2)
    d = d.rename(columns={
        'fiscal_quarter':'Quarter','lines':'Lines','po_count':'Orders',
        'spend_inr':'Spend ₹cr','on_time_rate':'On-time %','in_full_rate':'In-full %',
        'otif_rate':'OTIF %','avg_delay_days':'Avg delay',
        'stockouts':'Stockouts','invoice_error_rate':'Invoice errors %'})
    st.dataframe(d, hide_index=True, column_config={
        'Spend ₹cr': st.column_config.NumberColumn(format='%.2f'),
        'On-time %': st.column_config.NumberColumn(format='%.1f'),
        'In-full %': st.column_config.NumberColumn(format='%.1f'),
        'OTIF %': st.column_config.NumberColumn(format='%.1f'),
        'Avg delay': st.column_config.NumberColumn(format='%.1f'),
        'Invoice errors %': st.column_config.NumberColumn(format='%.1f')})

with t2:
    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_scatter(x=sq.fiscal_quarter, y=sq.external_credit_score, name='Credit score',
                        mode='lines+markers', line=dict(color=BLUE, width=2.5),
                        marker=dict(size=8),
                        hovertemplate='%{x}<br>Credit %{y:.1f}<extra></extra>')
        pr = sc.dropna(subset=['prob_rf'])
        if not pr.empty:
            fig.add_scatter(x=pr.fiscal_quarter, y=pr.prob_rf*100, name='Predicted risk %',
                            mode='lines+markers', line=dict(color=CRITICAL, width=2.5),
                            marker=dict(size=8),
                            hovertemplate='%{x}<br>Risk %{y:.1f}%<extra></extra>')
        fig.update_layout(title='Credit standing and predicted risk',
                          yaxis_title='0-100 scale', xaxis_title=None, yaxis_range=[0,105])
        fig.update_xaxes(categoryorder='array', categoryarray=D.quarters(), tickangle=-45)
        st.plotly_chart(style(fig, height=340), use_container_width=True)
    with r:
        fin = sq[['fiscal_quarter','altman_z_score','current_ratio','debt_to_equity_ratio']]
        fig = go.Figure()
        for c_, lab, colour in [('altman_z_score','Altman Z',BLUE),
                                ('current_ratio','Current ratio',AQUA),
                                ('debt_to_equity_ratio','Debt / equity',ORANGE)]:
            fig.add_scatter(x=fin.fiscal_quarter, y=fin[c_], name=lab, mode='lines+markers',
                            line=dict(width=2), marker=dict(size=7),
                            hovertemplate=f'{lab}<br>%{{x}}: %{{y:.2f}}<extra></extra>')
        fig.add_hline(y=1.8, line_dash='dash', line_color=CRITICAL, line_width=1.2)
        fig.add_annotation(x=0, y=1.8, text='Altman distress zone', showarrow=False,
                           xanchor='left', yshift=10, font=dict(size=10, color=CRITICAL))
        fig.update_layout(title='Financial ratios', yaxis_title='Ratio value', xaxis_title=None)
        fig.update_xaxes(categoryorder='array', categoryarray=D.quarters(), tickangle=-45)
        st.plotly_chart(style(fig, height=340), use_container_width=True)

    st.markdown('##### Statutory and market signals')
    sig = sq[['fiscal_quarter','gst_filing_delay_flag','litigation_case_count',
              'promoter_pledge_pct','news_sentiment_score','days_sales_outstanding']].rename(
        columns={'fiscal_quarter':'Quarter','gst_filing_delay_flag':'GST filing delayed',
                 'litigation_case_count':'Open cases','promoter_pledge_pct':'Promoter pledge %',
                 'news_sentiment_score':'News sentiment',
                 'days_sales_outstanding':'DSO (days)'})
    st.dataframe(sig, hide_index=True, column_config={
        'GST filing delayed': st.column_config.CheckboxColumn(),
        'Promoter pledge %': st.column_config.NumberColumn(format='%.0f'),
        'News sentiment': st.column_config.NumberColumn(format='%.2f'),
        'DSO (days)': st.column_config.NumberColumn(format='%.0f')})

with t3:
    if ev.empty:
        st.success('No incidents logged against this supplier in the selected period.')
    else:
        l, r = st.columns([1, 1.5])
        with l:
            cnt = ev.severity.value_counts().reindex(['Critical','High','Medium','Low']).fillna(0)
            fig = go.Figure(go.Bar(y=cnt.index, x=cnt.values, orientation='h',
                marker_color=[CRITICAL, SERIOUS, WARNING, MUTED],
                text=[int(v) for v in cnt.values], textposition='outside',
                textfont=dict(size=12, color=INK),
                hovertemplate='<b>%{y}</b><br>%{x} incidents<extra></extra>'))
            fig.update_layout(title='By severity', xaxis_title='Incidents', yaxis_title=None,
                              showlegend=False, xaxis_range=[0, max(cnt.values)*1.3])
            st.plotly_chart(style(fig, height=300), use_container_width=True)
        with r:
            t = ev.groupby(['fiscal_quarter','severity']).size().reset_index(name='n')
            fig = go.Figure()
            for sev, colour in [('Critical',CRITICAL),('High',SERIOUS),
                                ('Medium',WARNING),('Low',MUTED)]:
                g = t[t.severity == sev]
                if g.empty: continue
                fig.add_bar(x=g.fiscal_quarter, y=g.n, name=sev, marker_color=colour,
                            hovertemplate=f'{sev}<br>%{{x}}: %{{y}}<extra></extra>')
            fig.update_layout(barmode='stack', title='Incidents over time',
                              yaxis_title='Incidents', xaxis_title=None)
            fig.update_xaxes(categoryorder='array', categoryarray=D.quarters(), tickangle=-45)
            st.plotly_chart(style(fig, height=300), use_container_width=True)

        show = ev[['event_date','fiscal_quarter','event_type','severity','reported_by',
                   'resolved_flag','resolution_days']].sort_values('event_date',
                   ascending=False).rename(columns={
            'event_date':'Date','fiscal_quarter':'Quarter','event_type':'Type',
            'severity':'Severity','reported_by':'Reported by','resolved_flag':'Resolved',
            'resolution_days':'Days to resolve'})
        st.dataframe(show, hide_index=True, height=280, column_config={
            'Resolved': st.column_config.CheckboxColumn()})

with t4:
    peers = sup[sup.primary_category == s.primary_category]
    st.caption(f'Compared against {len(peers)} suppliers in {s.primary_category} '
               f'within the current filters.')
    metrics = [('otif_rate','OTIF rate', True), ('on_time_rate','On-time rate', True),
               ('invoice_accuracy','Invoice accuracy', True),
               ('external_credit_score','Credit score', True),
               ('avg_delay_days','Average delay', False),
               ('reject_rate_pct','Rejection rate', False)]
    rows = []
    for key, lab, hib in metrics:
        v = s[key]; pv = peers[key]
        pct = (pv < v).mean()*100 if hib else (pv > v).mean()*100
        rows.append(dict(Metric=lab, This=v, Median=pv.median(), Percentile=pct))
    comp = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(y=comp.Metric, x=comp.Percentile, orientation='h',
        marker_color=[GOOD if v>=50 else CRITICAL for v in comp.Percentile],
        text=[f'{v:.0f}th' for v in comp.Percentile], textposition='outside',
        textfont=dict(size=11, color=INK),
        hovertemplate='<b>%{y}</b><br>%{x:.0f}th percentile among peers<extra></extra>'))
    fig.add_vline(x=50, line_dash='dash', line_color=MUTED, line_width=1.2)
    fig.update_layout(title='Percentile against category peers (higher is better)',
                      xaxis_title='Percentile', yaxis_title=None, showlegend=False,
                      xaxis_range=[0,115])
    st.plotly_chart(style(fig, height=340), use_container_width=True)
    st.dataframe(comp, hide_index=True, column_config={
        'This': st.column_config.NumberColumn('This supplier', format='%.3f'),
        'Median': st.column_config.NumberColumn('Peer median', format='%.3f'),
        'Percentile': st.column_config.NumberColumn(format='%.0f')})
