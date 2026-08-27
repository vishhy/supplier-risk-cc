"""Shared visual language: navy/gold chrome, validated categorical palette for data marks."""
import plotly.graph_objects as go
import plotly.io as pio

# brand chrome
NAVY   = '#1F3864'
NAVY_D = '#16294A'
GOLD   = '#C8A227'
GOLD_L = '#E3C55A'
ICE    = '#F2F5FA'

# data marks: dataviz reference palette (validated for CVD separation and contrast)
BLUE   = '#2a78d6'
ORANGE = '#eb6834'
AQUA   = '#1baf7a'
YELLOW = '#eda100'
VIOLET = '#4a3aa7'
SERIES = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]

# status (reserved, never reused as a series colour)
GOOD    = '#0ca30c'
WARNING = '#fab219'
SERIOUS = '#ec835a'
CRITICAL= '#d03b3b'

INK   = '#0b0b0b'
INK2  = '#52514e'
MUTED = '#898781'
GRID  = '#e1e0d9'
SURF  = '#ffffff'

RISK_COLOURS = {'Critical':CRITICAL, 'High':SERIOUS, 'Moderate':WARNING, 'Low':GOOD}
KRALJIC_COLOURS = {'Strategic':CRITICAL, 'Bottleneck':ORANGE, 'Leverage':BLUE,
                   'Non-critical':'#c3c2b7'}
GRADE_COLOURS = {'A — Preferred':GOOD, 'B — Maintain':BLUE,
                 'C — Develop':WARNING, 'D — Exit / Replace':CRITICAL}

_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family='Segoe UI, Calibri, system-ui, sans-serif', size=13, color=INK),
    paper_bgcolor=SURF, plot_bgcolor=SURF,
    colorway=SERIES,
    title=dict(font=dict(size=15, color=NAVY), x=0, xanchor='left', pad=dict(b=12)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor='#c3c2b7',
               tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=INK2, size=12))),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor='#c3c2b7',
               tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=INK2, size=12))),
    legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1,
                font=dict(size=11, color=INK2), bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=10, r=10, t=48, b=10),
    hoverlabel=dict(bgcolor='white', bordercolor=GRID, font=dict(size=12, color=INK)),
    hovermode='closest',
)
pio.templates['scm'] = _template
pio.templates.default = 'scm'

def style(fig, height=None, legend_top=True, **kw):
    fig.update_layout(template='scm', **kw)
    if height: fig.update_layout(height=height)
    if not legend_top: fig.update_layout(legend=dict(orientation='v', y=1, x=1.02, xanchor='left'))
    return fig

def inr(v, decimals=1):
    """Format rupees in the Indian convention: crore above 1e7, lakh above 1e5."""
    v = float(v)
    sign = '-' if v < 0 else ''
    v = abs(v)
    if v >= 1e7:  return f'{sign}₹{v/1e7:,.{decimals}f} cr'
    if v >= 1e5:  return f'{sign}₹{v/1e5:,.{decimals}f} L'
    if v >= 1e3:  return f'{sign}₹{v/1e3:,.0f}k'
    return f'{sign}₹{v:,.0f}'

CSS = """
<style>
  .block-container {padding-top: 2.0rem; padding-bottom: 2rem; max-width: 1500px;}
  h1 {font-size: 1.85rem !important; color: #1F3864; font-weight: 700; letter-spacing: -.3px;}
  h2 {font-size: 1.22rem !important; color: #1F3864; font-weight: 700; margin-top: .4rem;}
  h3 {font-size: 1.02rem !important; color: #1F3864; font-weight: 700;}
  .sub {color:#5A6270; font-size:.95rem; margin:-.45rem 0 1.0rem 0;}
  .kpi {background:#F2F5FA; border-radius:12px; padding:16px 18px; height:100%;}
  .kpi.dark {background:#1F3864;}
  .kpi .lab {font-size:.76rem; text-transform:uppercase; letter-spacing:.6px;
             color:#5A6270; font-weight:600;}
  .kpi.dark .lab {color:#A8B4C8;}
  .kpi .val {font-size:1.85rem; font-weight:700; color:#1F3864; line-height:1.25; margin-top:2px;}
  .kpi.dark .val {color:#E3C55A;}
  .kpi .sub2 {font-size:.78rem; color:#5A6270; margin-top:3px;}
  .kpi.dark .sub2 {color:#A8B4C8;}
  .note {background:#FDF6E3; border-radius:10px; padding:13px 17px; font-size:.9rem;
         color:#1A1A1A; line-height:1.55;}
  .note b {color:#8A6200;}
  .info {background:#F2F5FA; border-radius:10px; padding:13px 17px; font-size:.9rem;
         color:#1A1A1A; line-height:1.55;}
  .info b {color:#1F3864;}
  .chip {display:inline-block; background:#1F3864; color:#fff; border-radius:20px;
         padding:2px 11px; font-size:.74rem; margin:2px 4px 2px 0;}
  div[data-testid="stMetricValue"] {font-size:1.5rem; color:#1F3864;}
  section[data-testid="stSidebar"] {background:#F7F9FC;}
  section[data-testid="stSidebar"] h2 {font-size:1.0rem !important;}
  hr {margin: .9rem 0; border-color:#e1e0d9;}
</style>
"""

def kpi(label, value, sub='', dark=False):
    cls = 'kpi dark' if dark else 'kpi'
    s = f'<div class="sub2">{sub}</div>' if sub else ''
    return (f'<div class="{cls}"><div class="lab">{label}</div>'
            f'<div class="val">{value}</div>{s}</div>')
