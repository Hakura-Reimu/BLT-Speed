cd ~/blt/code && source ~/blt/venv/bin/activate
python - <<'PY'
import torch, hashlib
from blt.config import preset
from blt.model import ByteLatentLM
def h(t): return hashlib.sha256(t.detach().float().contiguous().numpy().tobytes()).hexdigest()[:12]
ms={}
for n in ['S-P6','S-P6-poolbal-w256']:
    torch.manual_seed(0)
    ms[n]=ByteLatentLM(preset(n,min_len=3))
a,b=ms['S-P6'].state_dict(),ms['S-P6-poolbal-w256'].state_dict()
same=[k for k in a if k in b and a[k].shape==b[k].shape and torch.equal(a[k],b[k])]
diff=[k for k in a if k in b and a[k].shape==b[k].shape and not torch.equal(a[k],b[k])]
print('same-shape identical:',len(same),'same-shape different:',len(diff),'examples',diff[:4])
# corrected prefill MACs/byte at 8 KiB: global + encoder layer + pool + decoder tail (~64 positions) amortised
def layer(w,f): return 4*w*w+3*w*f
def gd(L,w,f): return L*(2*w*w+2*w*256+3*w*f)
T=8192
for name,G,P,w,f in [('S-P6',gd(6,512,1536),5.919,128,384),('S-P6-w256',gd(6,512,1536),5.919,256,768),('M-P8',gd(11,768,2048),7.734,192,512),('M-P8-w384',gd(11,768,2048),7.734,384,1024)]:
    gw=512 if name.startswith('S') else 768
    tail=64*(2*layer(w,f)+gw*w)/T
    print(name, round((G/P+layer(w,f)+w*gw/P+tail)/1e6,2))
print('S-BPE50K',round(gd(6,512,1536)/4.6246/1e6,2),'M-BPE50K',round(gd(9,768,2048)/4.6246/1e6,2))
PY
date
