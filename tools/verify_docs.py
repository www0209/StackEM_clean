#!/usr/bin/env python3
"""Re-check every number quoted in the documentation against the raw result files.

    python tools/verify_docs.py            # all docs/checks/*.jsonl
    python tools/verify_docs.py docs/checks/exp_03_rail_accuracy.jsonl

Each line of a checks file is one JSON object:
  doc     the markdown file that quotes the number
  claim   what the number is (free text)
  value   the number exactly as printed in the doc (after scaling), e.g. 0.26 for "0.26 %"
  digits  decimals printed (default: taken from value)
  scale   factor from the raw value to the printed one (default 1; 100 for per cent)
and ONE source:
  file + path   JSON file (repo-relative) and the list of keys / indices leading to the raw number
  file + regex  text or log file and a regular expression whose group 1 (or "group") is the raw number;
                "nth" picks the n-th match (default 0, -1 = last)
  py            a Python expression; J(path) loads a JSON file, T(path) reads a text file,
                median()/quantile()/np are available
A check passes when  |raw*scale - value| <= half a unit of the last printed digit.
"""
import json, re, sys, glob, os, math, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}
def J(path):
    p = os.path.join(ROOT, path)
    if p not in _cache:
        s = open(p, encoding="utf-8", errors="replace").read()
        s = re.sub(r'(?<![\w"])-?Infinity\b', lambda m: '"-inf"' if m.group(0).startswith('-') else '"inf"', s)
        s = re.sub(r'(?<![\w"])NaN\b', '"nan"', s)
        _cache[p] = json.loads(s)
    return _cache[p]
def T(path):
    return open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
def median(x): return statistics.median(x)
def quantile(x, q):
    x = sorted(x); k = (len(x) - 1) * q; f = math.floor(k); c = min(f + 1, len(x) - 1)
    return x[f] + (x[c] - x[f]) * (k - f)
class _Arr(list):
    """Minimal stand-in for a numpy array, used only when numpy is not installed."""
    def _bin(self, o, f):
        if isinstance(o, (list, tuple)): return _Arr(f(a, b) for a, b in zip(self, o))
        return _Arr(f(a, o) for a in self)
    def __sub__(self, o): return self._bin(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._bin(o, lambda a, b: b - a)
    def __add__(self, o): return self._bin(o, lambda a, b: a + b)
    __radd__ = __add__
    def __mul__(self, o): return self._bin(o, lambda a, b: a * b)
    __rmul__ = __mul__
    def __truediv__(self, o): return self._bin(o, lambda a, b: a / b)
    def __rtruediv__(self, o): return self._bin(o, lambda a, b: b / a)
    def __and__(self, o): return self._bin(o, lambda a, b: bool(a) and bool(b))
    def __or__(self, o): return self._bin(o, lambda a, b: bool(a) or bool(b))
    def __invert__(self): return _Arr(not a for a in self)
    def __neg__(self): return _Arr(-a for a in self)
    def __getitem__(self, k):
        if isinstance(k, (list, tuple)) and len(k) == len(self) and all(isinstance(x, bool) for x in k):
            return _Arr(a for a, m in zip(self, k) if m)
        r = list.__getitem__(self, k)
        return _Arr(r) if isinstance(k, slice) else r
class _MiniNumpy:
    array = staticmethod(lambda x: _Arr(x))
    asarray = array
    @staticmethod
    def abs(x): return _Arr(abs(a) for a in x) if isinstance(x, (list, tuple)) else abs(x)
    @staticmethod
    def isfinite(x): return _Arr(math.isfinite(a) for a in x) if isinstance(x, (list, tuple)) else math.isfinite(x)
    max = staticmethod(lambda x: max(x))
    min = staticmethod(lambda x: min(x))
    sum = staticmethod(lambda x: sum(x))
    mean = staticmethod(lambda x: sum(x) / len(x))
    median = staticmethod(lambda x: statistics.median(x))
    percentile = staticmethod(lambda x, q: quantile(list(x), q / 100.0))
try:
    if os.environ.get("NO_NUMPY"): raise ImportError
    import numpy as np
except Exception:
    np = _MiniNumpy()          # the checks only need the few functions above
def decimals(v):
    s = repr(v)
    if "e" in s or "E" in s: return None
    return len(s.split(".")[1]) if "." in s else 0
def raw_of(c):
    if "py" in c:
        return eval(c["py"], {"J": J, "T": T, "median": median, "quantile": quantile, "np": np, "math": math, "abs": abs, "len": len, "sum": sum, "min": min, "max": max, "float": float, "sorted": sorted, "round": round, "re": re})
    if "path" in c:
        v = J(c["file"])
        for k in c["path"]: v = v[k]
        return v
    ms = re.findall(c["regex"], T(c["file"]), flags=re.M)
    m = ms[c.get("nth", 0)]
    if isinstance(m, tuple): m = m[c.get("group", 1) - 1]
    return float(m)
def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "docs", "checks", "*.jsonl")))
    n = bad = 0
    for f in files:
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            line = line.strip()
            if not line or line.startswith("#"): continue
            n += 1
            try:
                c = json.loads(line); raw = raw_of(c)
                if isinstance(raw, str): raw = float(raw)
                shown = c["value"]; got = raw * c.get("scale", 1)
                if "rel_tol" in c:
                    ok = abs(got - shown) <= c["rel_tol"] * abs(shown)
                else:
                    d = c.get("digits", decimals(shown)); ok = abs(got - shown) <= 0.5 * 10 ** (-d) * 1.0000001
                if not ok:
                    bad += 1; print(f"FAIL {os.path.basename(f)}:{i}  {c.get('doc')}: {c.get('claim')}  doc={shown}  raw*scale={got!r}")
            except Exception as e:
                bad += 1; print(f"ERROR {os.path.basename(f)}:{i}  {type(e).__name__}: {e}")
    print(f"{n} checks, {n-bad} passed, {bad} failed")
    sys.exit(1 if bad else 0)
if __name__ == "__main__":
    main()
