# app/utils.py
import re
import numpy as np
import pandas as pd

_punc_re = re.compile(r"[^a-z0-9\s]")

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = _punc_re.sub(" ", s)
    return " ".join(s.split())

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def normalize_series(s: pd.Series):
    if s.isna().all():
        return s.fillna(0.0)
    return (s - s.mean()) / (s.std() + 1e-9)

# simple sentiment heuristic (very light): polarity by counting happy/sad tokens
POS_TOKENS = {"good","great","excellent","love","loved","best","amazing","nice","perfect","recommend"}
NEG_TOKENS = {"bad","terrible","worst","hate","hated","awful","poor","disappointed"}
def simple_sentiment(text: str) -> float:
    t = clean_text(text)
    if not t:
        return 0.0
    tokens = set(t.split())
    pos = len(tokens & POS_TOKENS)
    neg = len(tokens & NEG_TOKENS)
    score = (pos - neg) / (1 + pos + neg)
    return float(score)
