"""Mock Variant B: hidden neuron fires on z + sigma*||w*x||*eps; forward through the noisy neuron, STE on intended z."""
import gzip
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
def load(n, o): return np.frombuffer(gzip.open(DATA / n).read(), np.uint8, offset=o)
X = load("train-images-idx3-ubyte.gz",16).reshape(-1,784)/255.0; y = load("train-labels-idx1-ubyte.gz",8)
r = np.random.default_rng(0)
tr = np.concatenate([r.choice(np.where(y==d)[0],100,replace=False) for d in range(10)])
va = np.concatenate([r.choice(np.setdiff1d(np.where(y==d)[0],tr),100,replace=False) for d in range(10)])
sig = lambda z: 1/(1+np.exp(-np.clip(z,-30,30)))
def hid(W,b,Xb,s,rng):
    z = Xb@W+b
    if s: z = z + s*np.sqrt((Xb**2)@(W**2))*rng.standard_normal(z.shape)
    return z
def run(sig_train, sig_eval, seed, wd=0.0, ep=50):
    rng = np.random.default_rng(seed)
    P=[rng.normal(0,1/28,(784,32)),np.zeros(32),rng.normal(0,1/np.sqrt(32),(32,10)),np.zeros(10)]
    m=[0*p for p in P]; v=[0*p for p in P]; t=0; Y=np.eye(10)[y[tr]]
    for _ in range(ep):
        for i in np.array_split(rng.permutation(1000),31):
            Xb=X[tr][i]; zi=Xb@P[0]+P[1]; h=(hid(P[0],P[1],Xb,sig_train,rng)>0)*1.0
            z2=h@P[2]+P[3]; dz2=(sig(z2)-Y[i])/(10*len(i)); dz1=(dz2@P[2].T)*sig(zi)*(1-sig(zi))
            g=[Xb.T@dz1+wd*P[0],dz1.sum(0),h.T@dz2,dz2.sum(0)]; t+=1
            for k in range(4):
                m[k]=.9*m[k]+.1*g[k]; v[k]=.999*v[k]+.001*g[k]**2
                P[k]-=1e-2*(m[k]/(1-.9**t))/(np.sqrt(v[k]/(1-.999**t))+1e-8)
    ev = np.random.default_rng(99)
    accs=[(np.argmax(((hid(P[0],P[1],X[va],sig_eval,ev)>0)*1.0)@P[2]+P[3],1)==y[va]).mean() for _ in range(5)]
    zi=X[va]@P[0]+P[1]; agree=((hid(P[0],P[1],X[va],sig_eval,ev)>0)==(zi>0)).mean()
    return np.mean(accs), agree
for s in (0.5,1.0,2.0):
    for label,st in (("swap (trained clean)",0.0),("trained with noisy neuron",s)):
        res=np.array([run(st,s,seed) for seed in range(3)])
        print(f"sigma={s} {label:<27} val acc {res[:,0].mean():.3f} ±{res[:,0].std():.3f}  neuron agrees w/ true sign {res[:,1].mean():.3f}",flush=True)
