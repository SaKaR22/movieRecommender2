# model/recommender.py
from __future__ import annotations

import os, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.neighbors import NearestNeighbors
from difflib import SequenceMatcher


def _split_genres_series(s: pd.Series) -> List[List[str]]:
    if s is None: return [[]]
    out=[]
    for val in s.fillna("").astype(str).tolist():
        parts = re.split(r"[,\|;/]+", val)
        parts = [p.strip().lower() for p in parts if p.strip()]
        parts = [p.replace(" ", "") for p in parts]
        out.append(parts)
    return out

def _normalize_title(s: str) -> str:
    if s is None: return ""
    t = str(s).strip().lower()
    t = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _compose_text_row(row: pd.Series) -> str:
    bits=[]
    for col in ("soup","overview","tagline"):
        if col in row and pd.notna(row[col]): bits.append(str(row[col]))
    if "title" in row and pd.notna(row["title"]): bits.append(str(row["title"]))
    if "genres" in row and pd.notna(row["genres"]): bits.append(str(row["genres"]))
    for col in ("cast","actors","director","keywords","tags"):
        if col in row and pd.notna(row[col]): bits.append(str(row[col]))
    return " ".join(bits)


class MovieRecommender:
    def __init__(self,
                 csv_path: Optional[str] = None,
                 text_ngram: Tuple[int,int]=(1,2),
                 max_features: int = 50000,
                 min_df: int = 2,
                 genre_weight: float = 2.0,
                 nn_neighbors: int = 300):
        self.csv_path = csv_path or self._discover_csv()
        self.df = pd.read_csv(self.csv_path)
        if "title" not in self.df.columns:
            raise ValueError("CSV must contain a 'title' column")
        self.df["title"] = self.df["title"].fillna("").astype(str)

        if "genres" in self.df.columns:
            self.df["genres_list"] = _split_genres_series(self.df["genres"])
        else:
            self.df["genres_list"] = [[] for _ in range(len(self.df))]

        if "soup" in self.df.columns and self.df["soup"].notna().any():
            text_series = self.df["soup"].fillna("").astype(str)
        else:
            text_series = self.df.apply(_compose_text_row, axis=1).astype(str)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=text_ngram,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )
        X_text = self.vectorizer.fit_transform(text_series)

        self.mlb = MultiLabelBinarizer()
        G_dense = self.mlb.fit_transform(self.df["genres_list"])
        G = csr_matrix(G_dense) * float(genre_weight)

        X = hstack([X_text, G]).tocsr()
        self.X_norm = normalize(X, norm="l2", copy=False)

        n_nbrs = min(nn_neighbors, self.X_norm.shape[0]-1) if self.X_norm.shape[0] > 1 else 1
        self.nn = NearestNeighbors(n_neighbors=max(n_nbrs, 1), metric="cosine", algorithm="brute")
        self.nn.fit(self.X_norm)
        self._title_to_idx: Dict[str,int] = {_normalize_title(t): i for i,t in enumerate(self.df["title"])}

    def get_recommendations(self, title: str, k: int = 10) -> pd.DataFrame:
        if not title:
            return pd.DataFrame(columns=["title","score","genres","director"])

        i, _, s = self._resolve_index(title)

        if i is not None and s >= 0.55:
            idx = self.nn.kneighbors(self.X_norm[i], return_distance=False)[0]
            idx = [j for j in idx if j != i]
            scores = None 
            top_idx = idx[:k]
        else:
            q_text = self.vectorizer.transform([str(title)])
            zeros = csr_matrix((1, len(self.mlb.classes_)))
            q = hstack([q_text, zeros]).tocsr()
            q = normalize(q, norm="l2", copy=False)
            idx = self.nn.kneighbors(q, return_distance=False)[0]
            top_idx = idx[:k]
            scores = None

        rows=[]
        for j in top_idx:
            row = self.df.iloc[j]
            genres = ",".join(row.get("genres_list", []))
            rows.append({
                "title": str(row.get("title","")),
                "score": float("nan") if scores is None else float(scores[j]),
                "genres": genres,
                "director": str(row.get("director","")) if "director" in self.df.columns else "",
            })
        return pd.DataFrame(rows)

    def _discover_csv(self) -> str:
        here = Path(__file__).resolve().parent
        candidates = [
            here / ".." / "data" / "movies_processed.csv",
            here / "movies_processed.csv",
            Path(os.getcwd()) / "data" / "movies_processed.csv",
            Path(os.getcwd()) / "movies_processed.csv",
        ]
        for p in candidates:
            p = p.resolve()
            if p.exists(): return str(p)
        raise FileNotFoundError("couldn't find movies_processed.csv")

    def _resolve_index(self, raw_title: str) -> Tuple[Optional[int], Optional[str], float]:
        q = _normalize_title(raw_title)
        if q in self._title_to_idx:
            return self._title_to_idx[q], raw_title, 1.0

        best_i, best_s = None, 0.0
        for idx, t in enumerate(self.df["title"]):
            s = SequenceMatcher(a=q, b=_normalize_title(t)).ratio()
            if s > best_s:
                best_s, best_i = s, idx
        if best_i is not None:
            return best_i, self.df.iloc[best_i]["title"], float(best_s)
        return None, None, 0.0