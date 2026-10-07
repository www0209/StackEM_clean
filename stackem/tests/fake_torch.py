"""
A minimal numpy-backed stand-in for the subset of the torch API used by
stackem.closure_torch / skn_model, so that the torch code path can be
numerically checked on a machine without PyTorch (no autograd).

Usage (tests only):
    import stackem.tests.fake_torch as ft; ft.install()
    import stackem.closure_torch as ct     # now runs on numpy
"""
from __future__ import annotations
import sys, types, math
import numpy as np

float32 = np.float32; float64 = np.float64; long = np.int64


class device:
    def __init__(self, s="cpu"):
        self.type = str(s)

    def __repr__(self):
        return f"device({self.type})"


class Tensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    # --- properties -------------------------------------------------------
    @property
    def shape(self):
        return self.a.shape

    @property
    def dtype(self):
        return self.a.dtype

    @property
    def device(self):
        return device("cpu")

    def __len__(self):
        return len(self.a)

    def __repr__(self):
        return f"FakeTensor({self.a!r})"

    def numel(self):
        return self.a.size

    @property
    def is_cuda(self):
        return False

    # --- conversions ------------------------------------------------------
    def to(self, *args, **kw):
        for x in args:
            if x in (np.float32, np.float64, np.int64):
                return Tensor(self.a.astype(x))
        return self

    def float(self):
        return Tensor(self.a.astype(np.float64))

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def detach(self):
        return self

    def item(self):
        return float(self.a)

    def __float__(self):
        return float(self.a)

    def requires_grad_(self, *a):
        return self

    def clone(self):
        return Tensor(self.a.copy())

    def copy_(self, other):
        self.a[...] = _u(other); return self

    # --- shape ops --------------------------------------------------------
    def reshape(self, *s):
        if len(s) == 1 and isinstance(s[0], (tuple, list)): s = s[0]
        return Tensor(self.a.reshape(s))

    def expand(self, *s):
        return Tensor(np.broadcast_to(self.a, s))

    def unsqueeze(self, d):
        return Tensor(np.expand_dims(self.a, d))

    def max(self, dim=None):
        if dim is None:
            return Tensor(self.a.max())
        r = types.SimpleNamespace(values=Tensor(self.a.max(axis=dim)), indices=Tensor(self.a.argmax(axis=dim)))
        return r

    def min(self, dim=None):
        if dim is None:
            return Tensor(self.a.min())
        return types.SimpleNamespace(values=Tensor(self.a.min(axis=dim)), indices=Tensor(self.a.argmin(axis=dim)))

    def repeat_interleave(self, k, dim=0):
        return Tensor(np.repeat(self.a, k, axis=dim))

    def any(self, dim=None):
        return Tensor(self.a.any(axis=dim))

    def argmax(self, dim=None):
        return Tensor(self.a.argmax(axis=dim))

    # --- indexing ---------------------------------------------------------
    def __getitem__(self, k):
        return Tensor(self.a[_ui(k)])

    def __setitem__(self, k, v):
        self.a[_ui(k)] = _u(v)

    # --- arithmetic -------------------------------------------------------
    def __add__(self, o): return Tensor(self.a + _u(o))
    __radd__ = __add__
    def __sub__(self, o): return Tensor(self.a - _u(o))
    def __rsub__(self, o): return Tensor(_u(o) - self.a)
    def __mul__(self, o): return Tensor(self.a * _u(o))
    __rmul__ = __mul__
    def __truediv__(self, o): return Tensor(self.a / _u(o))
    def __rtruediv__(self, o): return Tensor(_u(o) / self.a)
    def __pow__(self, o): return Tensor(self.a ** _u(o))
    def __rpow__(self, o): return Tensor(_u(o) ** self.a)
    def __neg__(self): return Tensor(-self.a)
    def __matmul__(self, o): return Tensor(self.a @ _u(o))
    def __lt__(self, o): return Tensor(self.a < _u(o))
    def __gt__(self, o): return Tensor(self.a > _u(o))
    def __ge__(self, o): return Tensor(self.a >= _u(o))
    def __le__(self, o): return Tensor(self.a <= _u(o))
    def __eq__(self, o): return Tensor(self.a == _u(o))
    def __and__(self, o): return Tensor(self.a & _u(o))


def _u(x):
    return x.a if isinstance(x, Tensor) else x


def _ui(k):
    if isinstance(k, tuple):
        return tuple(_u(i) for i in k)
    return _u(k)


def tensor(a, dtype=None, device=None):
    return Tensor(np.asarray(a, dtype=dtype))


as_tensor = tensor


def zeros(*s, dtype=None, device=None):
    if len(s) == 1 and isinstance(s[0], (tuple, list)): s = s[0]
    return Tensor(np.zeros(s, dtype=dtype or np.float64))


def ones(*s, dtype=None, device=None):
    if len(s) == 1 and isinstance(s[0], (tuple, list)): s = s[0]
    return Tensor(np.ones(s, dtype=dtype or np.float64))


def full(s, v, dtype=None, device=None):
    return Tensor(np.full(s, v, dtype=dtype))


def zeros_like(t): return Tensor(np.zeros_like(_u(t)))
def ones_like(t): return Tensor(np.ones_like(_u(t)))
def full_like(t, v): return Tensor(np.full_like(_u(t), v))
def cat(ts, dim=0): return Tensor(np.concatenate([_u(t) for t in ts], axis=dim))
def stack(ts, dim=0): return Tensor(np.stack([_u(t) for t in ts], axis=dim))
def log(t): return Tensor(np.log(_u(t)))
def log10(t): return Tensor(np.log10(_u(t)))
def exp(t): return Tensor(np.exp(_u(t)))
def sqrt(t): return Tensor(np.sqrt(_u(t)))
def tanh(t): return Tensor(np.tanh(_u(t)))
def sin(t): return Tensor(np.sin(_u(t)))
def cos(t): return Tensor(np.cos(_u(t)))
def abs(t): return Tensor(np.abs(_u(t)))
def clamp(t, min=None, max=None): return Tensor(np.clip(_u(t), min, max))
def where(c, a, b): return Tensor(np.where(_u(c), _u(a), _u(b)))
def arange(n, device=None): return Tensor(np.arange(n))
def is_tensor(x): return isinstance(x, Tensor)
def manual_seed(s): np.random.seed(s)


def linspace(a, b, n, device=None, dtype=None): return Tensor(np.linspace(a, b, n))
def maximum(a, b): return Tensor(np.maximum(_u(a), _u(b)))
def minimum(a, b): return Tensor(np.minimum(_u(a), _u(b)))
def floor(t): return Tensor(np.floor(_u(t)))
def gather(t, dim, idx): return Tensor(np.take_along_axis(_u(t), _u(idx), dim))
def einsum(eq, *ts): return Tensor(np.einsum(eq, *[_u(t) for t in ts]))
int64 = np.int64
def diag_embed(t, offset=0):
    a = _u(t); n = a.shape[-1] + __builtins__['abs'](offset) if isinstance(__builtins__, dict) else a.shape[-1] + (offset if offset > 0 else -offset)
    out = np.zeros(a.shape[:-1] + (n, n), dtype=a.dtype)
    i = np.arange(a.shape[-1]); r = i - min(offset, 0); c = i + max(offset, 0)
    out[..., r, c] = a
    return Tensor(out)
linalg = types.SimpleNamespace(solve=lambda A, b: Tensor(np.linalg.solve(_u(A), _u(b))))


def tril_indices(n, m, device=None):
    r, c = np.tril_indices(n, 0, m)
    return Tensor(r), Tensor(c)


class no_grad:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Linear:
    def __init__(self, i, o):
        self.weight = Tensor(np.zeros((o, i), np.float32)); self.bias = Tensor(np.zeros(o, np.float32))

    def __call__(self, x):
        return Tensor(_u(x) @ self.weight.a.T + self.bias.a)


class _Module:
    def __init__(self):
        self._buffers = {}

    def register_buffer(self, name, t):
        setattr(self, name, t)

    def to(self, *a, **k): return self
    def eval(self): return self
    def train(self): return self

    def parameters(self):
        return []

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class _ModuleList(list):
    pass


class _Act:
    def __init__(self, f): self.f = f
    def __call__(self, x): return Tensor(self.f(_u(x)))


nn = types.SimpleNamespace(
    Module=_Module, Linear=_Linear, ModuleList=_ModuleList,
    SiLU=lambda: _Act(lambda x: x / (1 + np.exp(-x))), Tanh=lambda: _Act(np.tanh),
    GELU=lambda: _Act(lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))),
    init=types.SimpleNamespace(xavier_uniform_=lambda w: None, zeros_=lambda b: None),
)


def install():
    """Register this module as `torch` (and torch.nn) in sys.modules."""
    mod = sys.modules[__name__]
    mod.nn = nn
    mod.Tensor = Tensor
    sys.modules["torch"] = mod
    sys.modules["torch.nn"] = nn
    for k in list(sys.modules):
        if k.startswith("stackem.closure_torch") or k.startswith("stackem.skn_model"):
            del sys.modules[k]
