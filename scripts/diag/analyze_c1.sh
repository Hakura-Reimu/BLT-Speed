# C1 pilot, frozen-readout probe and S-BPE16K analysis (2026-11-10 08:21). Run inside WSL: bash -l < scripts/diag/analyze_c1.sh
cd ~/blt/results/main-scores
~/blt/venv/bin/python - <<'PY'
import json, numpy as np
LN2=np.log(2)
def load(n):
    d=json.load(open(n+'.json'))
    nats=np.array([c['nats'] for c in d['chunks']]); b=np.array([c['bytes'] for c in d['chunks']])
    return nats,b,[c['document'] for c in d['chunks']],d['bpb']
rng=np.random.default_rng(0)
def paired(a,bn,label):
    na,ba,da,pa=load(a); nb,bb,db,pb=load(bn)
    assert (ba==bb).all() and da==db
    ud={}
    for i,dd in enumerate(da): ud.setdefault(dd,[]).append(i)
    groups=list(ud.values())
    gna=np.array([na[g].sum() for g in groups]); gnb=np.array([nb[g].sum() for g in groups]); gb=np.array([ba[g].sum() for g in groups])
    diffs=[]
    for _ in range(2000):
        idx=rng.integers(0,len(groups),len(groups))
        diffs.append((gna[idx].sum()-gnb[idx].sum())/(LN2*gb[idx].sum()))
    lo,hi=np.percentile(diffs,[2.5,97.5])
    d=(na.sum()-nb.sum())/(LN2*ba.sum())
    print(f"{label:48s} {pa:.4f} vs {pb:.4f} diff={d:+.4f} 95%CI [{lo:+.4f},{hi:+.4f}]")
for m in ['bytes-375000000','bytes-750000000']:
    paired(f'pilot-s-p6-gread-{m}',f's-p6-seed0-{m}',f'gread - S-P6 @ {m}')
    paired(f'pilot-s-p6-gread-{m}',f'pilot-s-p6-poolbal-{m}',f'gread - poolbal @ {m}')
print("S-BPE16K")
for m in ['bytes-375000000','bytes-750000000','bytes-1500000000','bytes-3000000000','active-3600s']:
    paired(f's-bpe16k-seed0-{m}',f's-bpe50k-seed0-{m}',f'16K - 50K @ {m}')
    paired(f's-p6-seed0-{m}',f's-bpe16k-seed0-{m}',f'S-P6 - 16K @ {m}')
PY
~/blt/venv/bin/python - <<'PY'
import json, os
def rows(run): return [json.loads(l) for l in open(os.path.expanduser(f'~/blt/runs/{run}/log.jsonl'))]
r=rows('pilot-s-p6-gread'); print('keys', sorted(set().union(*[x.keys() for x in r])))
def vals(run): return {x['step']:x['val_bpb'] for x in rows(run) if x.get('val_bpb') is not None}
def train(run): return {x['step']:x['loss'] for x in rows(run) if 'loss' in x and x.get('step') is not None}
g=vals('pilot-s-p6-gread'); c=vals('confirm-s-p6-1.2e-03'); a=vals('pilot-s-p6-poolbal'); w=vals('pilot-s-p6-poolbal-w256')
print('in-training val (256 windows), gread minus reference')
for s in sorted(g):
    print(f"step {s:5d} gread={g[s]:.4f} -confirm={g[s]-c[s]:+.4f}" if s in c else f"step {s:5d} gread={g[s]:.4f} -confirm=   n/a ", f"-poolbal={g[s]-a[s]:+.4f}" if s in a else "", f"-w256={g[s]-w[s]:+.4f}" if s in w else "")
tm=train('s-p6-seed0'); tc=train('confirm-s-p6-1.2e-03'); tg=train('pilot-s-p6-gread')
common=sorted(set(tm)&set(tc))
import statistics
print('train-loss steps main&confirm', len(common), common[:3], common[-3:])
if common:
    d=[tm[s]-tc[s] for s in common]; print('main-confirm train loss: max|d|', max(map(abs,d)), 'first steps', [round(x,5) for x in d[:5]], 'last', [round(x,5) for x in d[-5:]])
cg=sorted(set(tm)&set(tg))
if cg:
    d=[tg[s]-tm[s] for s in cg]; print('gread-main train loss: first', [round(x,5) for x in d[:5]], 'windows of 200 mean', [round(statistics.mean(d[i:i+200]),4) for i in range(0,len(d),400)])
for run in ['pilot-s-p6-gread','pilot-s-p6-poolbal','pilot-s-p6-poolbal-w256']:
    rr=rows(run); last=[x for x in rr if 'active_seconds' in x or 'elapsed' in x]
    print(run, 'last row', {k:v for k,v in rr[-1].items() if not isinstance(v,(list,dict))})
PY
~/blt/venv/bin/python - <<'PY'
import json, os, statistics
def rows(run): return [json.loads(l) for l in open(os.path.expanduser(f'~/blt/runs/{run}/log.jsonl'))]
def train(run): return {x['step']:x['loss_nats_per_byte'] for x in rows(run) if 'loss_nats_per_byte' in x}
def speed(run): return {x['step']:x['bytes_per_second'] for x in rows(run) if 'bytes_per_second' in x}
tm=train('s-p6-seed0'); tc=train('confirm-s-p6-1.2e-03'); tg=train('pilot-s-p6-gread'); tp=train('pilot-s-p6-poolbal')
c=sorted(set(tm)&set(tc)); print('logged steps main&confirm', len(c), c[:4], c[-2:])
d=[tm[s]-tc[s] for s in c]; print('main-confirm loss diff (nats/byte): first', [round(x,5) for x in d[:6]], 'max|d|', round(max(map(abs,d)),5), 'mean|d| last 100', round(statistics.mean(map(abs,d[-100:])),5))
c=sorted(set(tm)&set(tg)); d=[tg[s]-tm[s] for s in c]
print('gread-main loss diff, first logged', [(s,round(tg[s]-tm[s],5)) for s in c[:4]])
for lo in (0,500,1000,2000,3000,4000,5000):
    seg=[tg[s]-tm[s] for s in c if lo<=s<lo+500]
    if seg: print(f'  steps {lo}-{lo+499}: mean {statistics.mean(seg)/0.6931:+.4f} bits/byte (n={len(seg)})')
sg=speed('pilot-s-p6-gread'); sp=speed('pilot-s-p6-poolbal'); sm=speed('s-p6-seed0')
for name,s in (('gread',sg),('poolbal',sp),('main',sm)):
    v=[s[k] for k in s if 1000<=k<=5700]; print(name,'median bytes/s steps 1000-5700', round(statistics.median(v)))
PY
