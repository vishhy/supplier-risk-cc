"""
Supplier Risk Command Centre
Entry point: page config, the global cross-filter rail, and navigation.
"""
import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title='Supplier Risk Command Centre', page_icon='📦',
                   layout='wide', initial_sidebar_state='expanded')

from lib import data as D
from lib.theme import CSS, NAVY, GOLD

st.markdown(CSS, unsafe_allow_html=True)
D.init_filters()

# ----------------------------------------------------------------- sidebar rail
with st.sidebar:
    st.markdown(
        f"<div style='padding:2px 0 10px 0'>"
        f"<div style='color:{NAVY};font-weight:700;font-size:1.05rem;line-height:1.25'>"
        f"Supplier Risk<br>Command Centre</div>"
        f"<div style='color:#5A6270;font-size:.78rem;margin-top:3px'>"
        f"RetailCo India · FY2023-24 to FY2025-26</div></div>",
        unsafe_allow_html=True)
    st.divider()

    s = D.suppliers(); q = D.quarters()
    st.markdown('## Filters')
    st.caption('Every chart, table and model output on every page respects these.')

    st.select_slider('Fiscal quarter range', options=q, value=(q[0], q[-1]), key='f_q')
    st.multiselect('Category', sorted(s.primary_category.unique()), key='f_cat',
                   placeholder='All categories')
    c1, c2 = st.columns(2)
    with c1:
        st.multiselect('Tier', ['Strategic','Preferred','Transactional'], key='f_tier',
                       placeholder='All')
        st.multiselect('ABC class', ['A','B','C'], key='f_abc', placeholder='All')
    with c2:
        st.multiselect('Region', sorted(s.region.unique()), key='f_region', placeholder='All')
        st.multiselect('Size', ['Large','Medium','Micro/Small'], key='f_size', placeholder='All')
    st.radio('Sourcing', ['All suppliers','Sole-sourced only','Has alternatives only'],
             key='f_sole', horizontal=False)

    n = len(D.supplier_ids())
    lo, hi = D.quarter_range()
    st.markdown(
        f"<div style='background:{NAVY};border-radius:10px;padding:11px 14px;margin-top:6px'>"
        f"<div style='color:#A8B4C8;font-size:.72rem;text-transform:uppercase;"
        f"letter-spacing:.6px;font-weight:600'>Current selection</div>"
        f"<div style='color:{GOLD};font-size:1.35rem;font-weight:700'>{n} suppliers</div>"
        f"<div style='color:#A8B4C8;font-size:.76rem'>{hi-lo+1} quarters in scope</div></div>",
        unsafe_allow_html=True)
    if n == 0:
        st.error('No suppliers match these filters. Widen the selection.')
    st.button('Reset all filters', on_click=D.reset_filters, width='stretch')

    st.divider()
    st.caption('Synthetic dataset built for this study. Figures are illustrative of method, '
               'not of any real supplier base.')

# ----------------------------------------------------------------- navigation
pages = [
    st.Page('views/overview.py',     title='Executive Overview',   icon=':material/dashboard:'),
    st.Page('views/performance.py',  title='Delivery & Invoicing', icon=':material/local_shipping:'),
    st.Page('views/scorecard.py',    title='Supplier Scorecard',   icon=':material/scoreboard:'),
    st.Page('views/prediction.py',   title='Risk Prediction',      icon=':material/online_prediction:'),
    st.Page('views/simulator.py',    title='What-If Simulator',    icon=':material/tune:'),
    st.Page('views/segmentation.py', title='Portfolio Strategy',   icon=':material/grid_view:'),
    st.Page('views/supplier360.py',  title='Supplier 360',         icon=':material/person_search:'),
]
st.navigation(pages).run()
