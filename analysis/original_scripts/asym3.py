
import sys
sys.path.insert(0, r"F:\MKV\_work\code_v5\markov-reversibility-code\src")
sys.path.insert(0, r"F:\MKV\_work")
import numpy as np
from scipy.optimize import minimize
from markov_research.models import reversible_matrix
PAIRS=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))
PI=np.ones(3)/3; diag=0.35/3; off=(1/3-diag)/2
S=np.array([[diag,off,off],[off,diag,off],[off,off,diag]]); S=S*((1/3)/S.sum(1,keepdims=True))
C=np.array([[0,1,-1],[-1,0,1],[1,-1,0]])
def build(J): return (S+(J/2)*C)/PI[:,None]
def sample_chain(P,n,rng):
    cum=np.cumsum(P,1); st=rng.integers(0,3); out=np.empty(n,dtype=np.int8); out[0]=st
    for t in range(1,n):
        u=rng.random(); st=(cum[st]<u).sum(); out[t]=st
    return out
def expl(P,Q): return float(-np.sum(PI*np.sum(P*np.log(Q),axis=1)))
def terms(st,a=1.0):
    Cm=np.zeros((3,3)); np.add.at(Cm,(st[:-1],st[1:]),1.0); sc=Cm+a
    return np.array([sc[i,i] if i==j else sc[i,j]+sc[j,i] for (i,j) in PAIRS]), sc.sum(1)
def nll(m,c,r):
    F=np.zeros((3,3)); pi=np.zeros(3)
    for k,(i,j) in enumerate(PAIRS):
        if i==j: F[i,i]=m[k]; pi[i]+=m[k]
        else: F[i,j]=m[k]/2; F[j,i]=m[k]/2; pi[i]+=m[k]/2; pi[j]+=m[k]/2
    return float(-np.sum(c*np.log(m))+np.sum(r*np.log(pi)))
def fit_slsqp(st,a=1.0):
    c,r=terms(st,a); cons=[{"type":"eq","fun":lambda m:m.sum()-1,"jac":lambda m:np.ones(6)}]
    best=None
    for x0 in (c/c.sum(), np.full(6,1/6)):
        res=minimize(nll,x0,args=(c,r),method="SLSQP",bounds=[(1e-12,None)]*6,constraints=cons,options={"maxiter":2000,"ftol":1e-15})
        if best is None or res.fun<best.fun: best=res
    m=best.x; F=np.zeros((3,3)); pi=np.zeros(3)
    for k,(i,j) in enumerate(PAIRS):
        if i==j: F[i,i]=m[k]; pi[i]+=m[k]
        else: F[i,j]=m[k]/2; F[j,i]=m[k]/2; pi[i]+=m[k]/2; pi[j]+=m[k]/2
    return F/pi[:,None]
rng=np.random.default_rng(13)
P=build(-0.014); st=sample_chain(P,200000,rng)
try: Q=reversible_matrix(st)[0]
except RuntimeError: Q=fit_slsqp(st)
print("J=-0.014 L(P,P)=%.6f L(P,P*)=%.6f asymptotic cost=%+.6f" % (expl(P,P), expl(P,Q), expl(P,Q)-expl(P,P)))
