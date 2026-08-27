"""
Build the app's data layer: Parquet tables pre-aggregated so nothing heavy happens at runtime,
plus a fitted model pipeline saved to disk. Run once before deploying.
"""
import pandas as pd, numpy as np, json, os, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SRC = '/home/claude/scm/data/'
OUT = '/home/claude/scm/app/data/'
MOD = '/home/claude/scm/app/models/'
os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)

supplier = pd.read_csv(SRC+'dim_supplier.csv')
item     = pd.read_csv(SRC+'dim_item.csv')
contract = pd.read_csv(SRC+'dim_contract.csv')
po       = pd.read_csv(SRC+'fact_purchase_order.csv')
receipt  = pd.read_csv(SRC+'fact_goods_receipt.csv')
invoice  = pd.read_csv(SRC+'fact_invoice.csv')
risk     = pd.read_csv(SRC+'dim_supplier_quarterly_risk.csv')
event    = pd.read_csv(SRC+'fact_supplier_event.csv')
panel    = pd.read_csv(SRC+'ml_supplier_quarter_panel.csv')

QUARTERS = sorted(po.fiscal_quarter.unique(), key=lambda s:(s[2:7], s[-1]))
QIDX = {q:i for i,q in enumerate(QUARTERS)}
json.dump({'quarters':QUARTERS}, open(OUT+'meta.json','w'))

# ---------------------------------------------------------------- supplier summary
rec = receipt.merge(po[['purchase_order_id','fiscal_quarter','po_value_inr','order_priority']],
                    on='purchase_order_id')
rec['otif'] = rec.on_time_flag * rec.in_full_flag

deliv = rec.groupby('supplier_id', as_index=False).agg(
    lines=('goods_receipt_id','count'), on_time_rate=('on_time_flag','mean'),
    in_full_rate=('in_full_flag','mean'), otif_rate=('otif','mean'),
    avg_delay_days=('delay_days','mean'), p90_delay_days=('delay_days',lambda s:s.quantile(.9)),
    max_delay_days=('delay_days','max'), stockouts=('stockout_triggered_flag','sum'),
    rej=('rejected_quantity','sum'), recv=('received_quantity','sum'))
deliv['reject_rate_pct'] = deliv.rej/deliv.recv*100
deliv = deliv.drop(columns=['rej','recv'])

inv = invoice.groupby('supplier_id', as_index=False).agg(
    invoices=('invoice_id','count'),
    invoice_accuracy=('invoice_error_flag', lambda s: 1-s.mean()),
    dispute_rate=('dispute_raised_flag','mean'),
    avg_days_to_payment=('days_to_payment','mean'),
    abs_variance_inr=('invoice_variance_inr', lambda s: s.abs().sum()))

spend = po.groupby('supplier_id', as_index=False).agg(
    total_spend_inr=('po_value_inr','sum'), po_count=('purchase_order_id','count'),
    emergency_pos=('order_priority', lambda s:int((s=='Emergency').sum())))

sla = contract.groupby('supplier_id', as_index=False).agg(
    sla_target_pct=('sla_on_time_delivery_pct','mean'),
    committed_value_inr=('committed_annual_value_inr','sum'),
    penalty_pct=('penalty_clause_pct','mean'))

risk_ord = risk.assign(qi=risk.fiscal_quarter.map(QIDX)).sort_values(['supplier_id','qi'])
latest = risk_ord.groupby('supplier_id').tail(1)[
    ['supplier_id','fiscal_quarter','external_credit_score','credit_rating','altman_z_score',
     'days_sales_outstanding','debt_to_equity_ratio','current_ratio','promoter_pledge_pct',
     'news_sentiment_score','litigation_case_count']].rename(
    columns={'fiscal_quarter':'risk_as_of'})

ev = event.groupby('supplier_id', as_index=False).agg(
    events=('event_id','count'),
    critical_events=('severity', lambda s:int(s.isin(['High','Critical']).sum())))

cat_crit = item.groupby('item_category', as_index=False).criticality_score.mean().rename(
    columns={'item_category':'primary_category','criticality_score':'category_criticality'})

sup = (supplier.merge(spend,on='supplier_id').merge(deliv,on='supplier_id')
                .merge(inv,on='supplier_id').merge(sla,on='supplier_id',how='left')
                .merge(latest,on='supplier_id',how='left').merge(ev,on='supplier_id',how='left')
                .merge(cat_crit,on='primary_category',how='left'))
sup[['events','critical_events']] = sup[['events','critical_events']].fillna(0)
sup['spend_share_pct'] = sup.total_spend_inr/sup.total_spend_inr.sum()*100
sup['sla_gap_pp'] = sup.on_time_rate*100 - sup.sla_target_pct
sup = sup.sort_values('total_spend_inr', ascending=False).reset_index(drop=True)
sup['spend_rank'] = np.arange(1, len(sup)+1)
sup['cum_spend_pct'] = sup.spend_share_pct.cumsum()
sup['abc_class'] = np.where(sup.cum_spend_pct<=70,'A',np.where(sup.cum_spend_pct<=90,'B','C'))
sup.to_parquet(OUT+'supplier_summary.parquet', index=False)

# ---------------------------------------------------------------- quarterly rollups
qs = rec.groupby(['supplier_id','fiscal_quarter'], as_index=False).agg(
    lines=('goods_receipt_id','count'), on_time_rate=('on_time_flag','mean'),
    in_full_rate=('in_full_flag','mean'), otif_rate=('otif','mean'),
    avg_delay_days=('delay_days','mean'), stockouts=('stockout_triggered_flag','sum'))
# attribute each invoice to the fiscal quarter of the ORDER it bills, not the invoice date.
# An invoice raised in April against a March order otherwise creates a spurious 13th quarter,
# and analytically the accuracy belongs to the order period anyway.
inv_q = invoice.drop(columns=['fiscal_quarter']).merge(
    po[['purchase_order_id','fiscal_quarter']], on='purchase_order_id')
qi = inv_q.groupby(['supplier_id','fiscal_quarter'], as_index=False).agg(
    invoice_error_rate=('invoice_error_flag','mean'), dispute_rate=('dispute_raised_flag','mean'))
qp = po.groupby(['supplier_id','fiscal_quarter'], as_index=False).agg(
    spend_inr=('po_value_inr','sum'), po_count=('purchase_order_id','count'))
sq = qs.merge(qi,on=['supplier_id','fiscal_quarter'],how='outer') \
       .merge(qp,on=['supplier_id','fiscal_quarter'],how='outer') \
       .merge(risk,on=['supplier_id','fiscal_quarter'],how='left')
sq = sq[sq.fiscal_quarter.isin(QUARTERS)]
sq['qi'] = sq.fiscal_quarter.map(QIDX)
sq = sq.merge(supplier[['supplier_id','supplier_name','supplier_tier','business_size',
                        'primary_category','region','state']], on='supplier_id')
sq.to_parquet(OUT+'supplier_quarter.parquet', index=False)

# category x quarter for the descriptive page
cq = (po.merge(supplier[['supplier_id','primary_category']],on='supplier_id')
        .groupby(['primary_category','fiscal_quarter'],as_index=False)
        .agg(spend_inr=('po_value_inr','sum'), po_count=('purchase_order_id','count')))
cq2 = (rec.merge(supplier[['supplier_id','primary_category']],on='supplier_id')
         .groupby(['primary_category','fiscal_quarter'],as_index=False)
         .agg(on_time_rate=('on_time_flag','mean'), otif_rate=('otif','mean'),
              stockouts=('stockout_triggered_flag','sum')))
cq.merge(cq2,on=['primary_category','fiscal_quarter'],how='left').to_parquet(
    OUT+'category_quarter.parquet', index=False)

event.merge(supplier[['supplier_id','supplier_name','supplier_tier','primary_category','region']],
            on='supplier_id').to_parquet(OUT+'events.parquet', index=False)
invoice.groupby(['supplier_id','invoice_error_type'],as_index=False).size().to_parquet(
    OUT+'invoice_errors.parquet', index=False)
contract.to_parquet(OUT+'contracts.parquet', index=False)

# ---------------------------------------------------------------- model + predictions
T='target_delivery_failure_next_q'
panel['qi'] = panel.fiscal_quarter.map(QIDX)
CAT=['supplier_tier','business_size','primary_category','region']
EXCL=[T,'target_supplier_exit_next_q','target_disengaged_next_q','supplier_id',
      'fiscal_quarter','credit_rating','qi']
NUM=[c for c in panel.select_dtypes(include=np.number).columns if c not in EXCL]

def build(model):
    pre = ColumnTransformer([
        ('n', Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]), NUM),
        ('c', OneHotEncoder(handle_unknown='ignore', drop='first'), CAT)])
    return Pipeline([('pre',pre),('clf',model)])

RF = lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features='sqrt',
                                    class_weight='balanced', random_state=42, n_jobs=-1)
LR = lambda: LogisticRegression(max_iter=3000, class_weight='balanced', C=0.5)

# honest out-of-sample probabilities for every scoreable quarter (expanding window)
oos = []
for t in range(6, panel.qi.max()+1):
    tr, te = panel[panel.qi<t], panel[panel.qi==t]
    for name, mk in [('rf',RF),('lr',LR)]:
        m = build(mk()); m.fit(tr[NUM+CAT], tr[T])
        d = te[['supplier_id','fiscal_quarter',T]].copy()
        d['model']=name; d['prob']=m.predict_proba(te[NUM+CAT])[:,1]
        oos.append(d)
oos = pd.concat(oos)
oos = oos.pivot_table(index=['supplier_id','fiscal_quarter',T], columns='model',
                      values='prob').reset_index().rename(
    columns={'rf':'prob_rf','lr':'prob_lr'})
# single-column benchmark, min-max scaled within quarter so it is comparable to a probability
bench = panel[['supplier_id','fiscal_quarter','rolling_2q_on_time_rate']].copy()
bench['prob_benchmark'] = bench.groupby('fiscal_quarter').rolling_2q_on_time_rate.transform(
    lambda s: (s.max()-s)/(s.max()-s.min()) if s.max()>s.min() else 0.5)
oos = oos.merge(bench[['supplier_id','fiscal_quarter','prob_benchmark']],
                on=['supplier_id','fiscal_quarter'], how='left')

scored = panel.merge(oos.drop(columns=[T]), on=['supplier_id','fiscal_quarter'], how='left')
scored = scored.merge(supplier[['supplier_id','supplier_name','city','state',
                                'is_sole_source','num_alternate_suppliers','msme_registered',
                                'iso_certified','onboarding_date']].drop(
    columns=[c for c in ['is_sole_source','num_alternate_suppliers','msme_registered',
                         'iso_certified'] if c in panel.columns]), on='supplier_id', how='left')
scored = scored.merge(cat_crit, on='primary_category', how='left')
scored.to_parquet(OUT+'scored_panel.parquet', index=False)

# final model fitted on everything up to the last scoreable quarter, for the what-if simulator
final = build(RF()); final.fit(panel[NUM+CAT], panel[T])
joblib.dump({'pipeline':final,'num':NUM,'cat':CAT}, MOD+'rf_pipeline.joblib', compress=3)

o = oos.dropna(subset=['prob_rf'])
metrics = dict(
    auc_rf=float(roc_auc_score(o[T], o.prob_rf)),
    auc_lr=float(roc_auc_score(o[T], o.prob_lr)),
    auc_benchmark=float(roc_auc_score(o[T], o.prob_benchmark)),
    n_oos=int(len(o)), base_rate=float(o[T].mean()),
    quarters_scored=sorted(o.fiscal_quarter.unique(), key=lambda s:(s[2:7],s[-1])))
imp = pd.Series(final.named_steps['clf'].feature_importances_,
    index=NUM+list(final.named_steps['pre'].named_transformers_['c']
                   .get_feature_names_out(CAT))).sort_values(ascending=False)
imp.head(25).rename('importance').reset_index().rename(
    columns={'index':'feature'}).to_parquet(OUT+'feature_importance.parquet', index=False)
json.dump(metrics, open(OUT+'model_metrics.json','w'), indent=1)

for f in sorted(os.listdir(OUT)):
    print(f, round(os.path.getsize(OUT+f)/1024,1), 'KB')
print('model', round(os.path.getsize(MOD+'rf_pipeline.joblib')/1024/1024,2), 'MB')
print(json.dumps(metrics, indent=1))
