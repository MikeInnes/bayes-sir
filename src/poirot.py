# A simple PPL

import jax, statistics, collections
import jax.numpy as np
from jax import random, vmap
from jax.scipy.special import logsumexp
from optimisers import adam
from jax.interpreters.xla import DeviceArray

# Distributions

class Normal:
    def __init__(self, loc, scale):
        self.loc = loc + 0*scale
        self.scale = scale
    def logpdf(self, x):
        return jax.scipy.stats.norm.logpdf(x, self.loc, self.scale)
    def rsample(self, rng):
        return random.normal(rng, np.shape(self.loc))*self.scale + self.loc

class MultiNormal:
    def __init__(self, μ, Σ):
        self.μ = μ
        self.Σ = Σ

    def logpdf(self, x):
        if x.ndim == 2:
            return vmap(self.logpdf)(x).sum()
        C = np.linalg.cholesky(self.Σ)
        dist = np.sum(np.linalg.solve(C, x - self.μ)**2)
        det = np.linalg.slogdet(self.Σ)[1]
        return -((len(x) * np.log(2*np.pi) + det) + dist)/2

def BinomialApprox(n, p, scale = 1):
    return Normal(n*p, scale * np.sqrt(np.clip(n, 1)*p*(1-p)))

def PoissonApprox(λ, scale = 1):
    return Normal(λ, scale * np.sqrt(np.clip(λ, 1)))

class RBF:
    def __init__(self, var = 1, h = 1):
        self.var = var
        self.h = h

    def apply(self, a, b):
        return self.var * np.exp(-np.sum(((a-b)/self.h)**2/2))

class Matern52:
    def __init__(self, var = 1, h = 1):
        self.var = var
        self.h = h
    def apply(self, a, b):
        d = np.abs(a - b) / self.h
        λ = np.sqrt(5) * d
        return self.var * (1 + λ + λ**2 / 3) * np.exp(-λ)

def cov(k, xs):
    return jax.vmap(lambda x: jax.vmap(lambda y: k.apply(x, y))(xs))(xs)

class GaussianProcess:
    def __init__(self, kernel):
        self.kernel = kernel

    def at(self, x, noise = 0):
        Σ = cov(self.kernel, x) + np.identity(x.shape[0])*noise
        return MultiNormal(np.zeros(x.shape), Σ)

# Functor/pytree

def functor(x):
    if isinstance(x, list):
        return x, lambda xs: list(xs)
    elif isinstance(x, dict):
        ks, vs = zip(*sorted(x.items()))
        return vs, lambda vs: dict(zip(ks, vs))
    else:
        return [], lambda _: x

def isleaf(x):
    return len(functor(x)[0]) == 0

def fmap1(f, x, *xs):
    items, re = functor(x)
    rest = map(lambda x: functor(x)[0], xs)
    return re(map(f, items, *rest))

def fmap(f, *xs):
    if any(map(isleaf, xs)):
        return f(*xs)
    else:
        return fmap1(lambda *xs: fmap(f, *xs), *xs)

def ffold(f, x):
    if isleaf(x):
        return x
    else:
        items, re = functor(x)
        items = map(lambda x: ffold(f, x), items)
        return f(*items)

def fsum(x):
    return ffold(lambda *x: sum(x), x)

def ffirst(x):
    if isleaf(x):
        return x
    else:
        items, _ = functor(x)
        return ffirst(items[0])

def findex(xs, i):
    def f(x):
        return x[i, ...]
    return fmap(f, xs)

def fsize(xs):
    def f(x):
        return x.size
    return fsum(fmap(f, xs))

# Log likelihood DSL

lls = []

def dist(x, d):
    if len(lls) > 0:
        lls[-1] += d.logpdf(x).sum()

def loglikelihood(f, *args):
    lls.append(0)
    f(*args)
    return lls.pop()

def check():
    if np.isnan(lls[-1]) or np.isinf(lls[-1]):
        raise Exception("LL check failed.")

def each(d, f):
    cases = d.enumerate_support()
    lpdf = np.zeros(len(cases))
    for i in range(len(cases)):
        lls.append(0)
        dist(cases[i], d)
        f(cases[i])
        lpdf[i] = lls.pop()
    lls[-1] += logsumexp(lpdf, 0)

# Empirical densities

def hypot2(x, y):
    def f(x, y):
        return np.sum((x-y)**2)
    return fsum(fmap(f, x, y))

def logdensity(x, y):
    return -fsize(x)/2*np.log(hypot2(x, y))

def logmean(x):
    return logsumexp(x) - np.log(x.size)

def potentiali(xs, i):
    N = ffirst(xs).shape[0]
    if N <= 1:
        return 0
    j = np.arange(N - 1) + (np.arange(N - 1) >= i)
    x = findex(xs, i)
    xs = findex(xs, j)
    return logmean(vmap(logdensity, (None, 0))(x, xs))

def interaction(xs):
    N = ffirst(xs).shape[0]
    return jax.vmap(lambda i: potentiali(xs, i))(np.arange(0, N))

def energy(ll, xs):
    return (interaction(xs) - jax.vmap(ll)(xs)).mean()

# Inference routines

def fsample(rng, xs, n):
    K = ffirst(xs).shape[0]
    i = random.randint(rng, [n], 0, K)
    return findex(xs, i)

def init_samples(rng, x, K = 10):
    def f(x):
        nonlocal rng
        rng, key = random.split(rng)
        x = x[None,...] if isinstance(x, DeviceArray) else x
        x = np.repeat(x, K, 0)
        return x + random.normal(key, x.shape)/100
    return fmap(f, x)

class InferenceState:
    def __init__(self, x, K = 100):
        self.rng = random.PRNGKey(0)
        self.rng, key = random.split(self.rng)
        self.t = 0 # time step
        self.K = K
        self.xs = init_samples(key, x, K)

    def mean(self):
        def f(x):
            return x.mean(0)
        return fmap(f, self.xs)

    def sample(self):
        self.rng, key = random.split(self.rng)
        i = random.randint(key, [], 0, self.K)
        return findex(self.xs, i)

    def samples(self):
        return self.xs

    def quantile(self, q):
        def f(x):
            return np.quantile(x, q, 0)
        return fmap(f, self.xs)

def energy(model, xs):
    def ll(x):
        return loglikelihood(model, x)
    return (interaction(xs) - jax.vmap(ll)(xs)).mean()

def windowlen(n, b = 1):
    return n if b == None else max(n // b, 1)

def infer(model, state, unroll = 1, eta = 0.001, jit = True):
    init, update, params = adam(eta)
    opt = init(state.xs)
    window = collections.deque(maxlen=windowlen(5000, unroll))
    def step(rng, opt, t):
        keys = random.split(rng, unroll)
        for i in range(0, unroll):
            xs = params(opt)
            L, dxs = jax.value_and_grad(energy, argnums=1)(model, xs)
            opt = update(t+i, dxs, opt)
        return L, opt
    if jit:
        step = jax.jit(step)
    try:
        while True:
            state.rng, key = random.split(state.rng)
            L, opt = step(key, opt, state.t)
            if np.isnan(L) or np.isinf(L):
                raise Exception("Invalid loss")
            state.xs = params(opt)
            state.t += unroll
            window.append(-L.item())
            print("\r[ %e, %.2e iterations   " % (statistics.mean(window), state.t), end = "")
    except KeyboardInterrupt:
        pass
    return state
