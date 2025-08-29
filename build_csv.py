import argparse, json, re, sys, ast
from pathlib import Path
import pandas as pd

def _load_csv(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"Missing: {p}")
    return pd.read_csv(p, low_memory=False)

def _json_list(s):
    if isinstance(s, list): 
        return s
    if not isinstance(s, str):
        return []
    try:
        return json.loads(s)
    except Exception:
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []

def _names_from_json(s, key="name", top=None):
    arr = _json_list(s)
    out = [str(d.get(key, "")).strip() for d in arr if isinstance(d, dict) and d.get(key)]
    return out[:top] if (top and top > 0) else out

def _first_director(crew_json):
    for p in _json_list(crew_json):
        if p.get("job") == "Director" and p.get("name"):
            return p["name"]
    return None

def _tok(x, squash_space=True):
    s = str(x).strip().lower()
    if squash_space:
        s = s.replace(" ", "")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

def _ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            pass
        return [t.strip() for t in x.split(",") if t.strip()]
    return []

def _join_csv_cells(lst):
    """Join list into CSV cell safely."""
    lst = _ensure_list(lst)
    return ",".join([str(v).strip() for v in lst if str(v).strip()])

def _build_soup(gen_tokens, director_name, cast_names, kw_tokens):
    gen_list = _ensure_list(gen_tokens)
    kw_list  = _ensure_list(kw_tokens)
    cast_list = _ensure_list(cast_names)

    gen_norm = [_tok(t, True) for t in gen_list if t]
    kw_norm  = [_tok(t, True) for t in kw_list if t]
    cast_norm = [_tok(n, False) for n in cast_list if n]

    dir_tok = _tok(director_name or "", True) if director_name else ""

    parts = []
    parts += gen_norm
    parts += kw_norm
    parts += cast_norm
    if dir_tok:
        parts.append(dir_tok)

    seen, out = set(), []
    for t in parts:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return " ".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to Kaggle 'The Movies Dataset' folder")
    ap.add_argument("--out", default="data/movies_processed.csv", help="Output CSV path")
    ap.add_argument("--top-cast", type=int, default=8, help="Max cast names to include in soup")
    args = ap.parse_args()

    src = Path(args.src)
    md = _load_csv(src / "movies_metadata.csv")
    cr = _load_csv(src / "credits.csv")
    kw = _load_csv(src / "keywords.csv")

    for df in (md, cr, kw):
        df["id"] = pd.to_numeric(df["id"], errors="coerce")

    md = md[["id","title","genres","popularity","vote_count"]].dropna(subset=["id","title"]).copy()
    md["genres_list"] = md["genres"].apply(lambda s: [_tok(n, True) for n in _names_from_json(s, "name") if n])

    cr = cr.dropna(subset=["id"]).copy()
    cr["director"]   = cr["crew"].apply(_first_director)
    cr["cast_names"] = cr["cast"].apply(lambda s: _names_from_json(s, "name", top=args.top_cast))

    kw = kw.dropna(subset=["id"]).copy()
    kw["keywords_list"] = kw["keywords"].apply(lambda s: [_tok(n, True) for n in _names_from_json(s, "name") if n])

    # join
    df = (md
          .merge(cr[["id","director","cast_names"]], on="id", how="left")
          .merge(kw[["id","keywords_list"]], on="id", how="left"))

    for col in ("popularity","vote_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = (df.sort_values(by=["title","popularity","vote_count"], ascending=[True, False, False])
            .drop_duplicates(subset=["title"], keep="first")
            .reset_index(drop=True))

    # final 6 columns
    out = pd.DataFrame()
    out["title"]    = df["title"].astype(str)
    out["genres"]   = df["genres_list"].apply(lambda lst: _join_csv_cells(lst))
    out["director"] = df["director"].fillna("").astype(str)
    out["cast"]     = df["cast_names"].apply(lambda lst: _join_csv_cells(lst))
    out["keywords"] = df["keywords_list"].apply(lambda lst: _join_csv_cells(lst))
    out["soup"]     = df.apply(lambda r: _build_soup(
        r.get("genres_list"),
        r.get("director"),
        r.get("cast_names"),
        r.get("keywords_list")
    ), axis=1)

    out = out[(out["title"].str.len() > 0) & (out["soup"].str.len() > 0)].copy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows to {args.out}")

if __name__ == "__main__":
    main()