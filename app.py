from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, url_for
from model.recommender import MovieRecommender
import os, re, logging, requests, hashlib, ast
import numpy as np
import pandas as pd
from collections import Counter
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from dotenv import load_dotenv
APP_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=APP_DIR / ".env", override=False)

app = Flask(__name__)

SIM_DIR = Path('model/cosine_sim.npy').resolve().parent
CSV_DIR = Path('data/movies_processed.csv').resolve().parent
CSV_PATH = CSV_DIR / "movies_processed.csv"
SIM_PATH = SIM_DIR / "cosine_sim.npy"

logging.basicConfig(level=logging.INFO)

_df_cache = None
_sim_cache = None
_title_idx_exact = None
_title_idx_norm  = None

def load_movies_df():
    global _df_cache
    if _df_cache is None:
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        df.columns = [str(c).strip().lower() for c in df.columns]
        _df_cache = df
    return _df_cache

def load_cosine_sim():
    global _sim_cache
    if _sim_cache is None:
        _sim_cache = np.load(SIM_PATH) if SIM_PATH.exists() else None
    return _sim_cache

def _norm_title(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _build_title_indices():
    global _title_idx_exact, _title_idx_norm
    if _title_idx_exact is None or _title_idx_norm is None:
        df = load_movies_df()
        titles = df["title"].fillna("").astype(str).tolist()
        _title_idx_exact = {t.strip().lower(): i for i, t in enumerate(titles)}
        _title_idx_norm  = {_norm_title(t): i for i, t in enumerate(titles)}
    return _title_idx_exact, _title_idx_norm

def title_to_index(title: str):
    if not title:
        return None
    exact, norm = _build_title_indices()
    t_exact = title.strip().lower()
    if t_exact in exact:
        return exact[t_exact]
    tn = _norm_title(title)
    return norm.get(tn)

def resolve_title(query: str):
    df = load_movies_df()
    exact, norm = _build_title_indices()
    idx = title_to_index(query)
    if idx is None:
        return None
    return str(df["title"].iloc[idx])

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_STOP = {"the","and","a","an","of","in","on","for","to","with","by","from","at","as",
         "is","it","this","that","he","she","they","we","you","be","are","was","were","or","not"}

def _tokenize_soup(s: str, top_k=50):
    if pd.isna(s) or not str(s).strip():
        return Counter()
    toks = [t.lower() for t in _WORD_RE.findall(str(s))]
    toks = [t for t in toks if len(t) > 2 and t not in _STOP]
    return Counter(toks).most_common(top_k)

def sample_similarity_from_soup(df: pd.DataFrame, n: int = 30):
    if "soup" not in df.columns: 
        return None
    s = df["soup"].fillna("").astype(str).head(n)
    if s.empty:
        return None
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(s)
    sim = cosine_similarity(X)
    return sim.round(4).tolist()

def _year(v):
    if pd.isna(v): 
        return None
    s = str(v)
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return int(m.group(0)) if m else None

def _norm_terms(val, *, squash_spaces=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()

    if isinstance(val, (list, tuple, set, np.ndarray)):
        items = list(val)
    else:
        s = str(val).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set, np.ndarray)):
                    items = list(parsed)
                else:
                    items = [parsed]
            except Exception:
                items = re.split(r"[,\|;/]+", s)
        else:
            items = re.split(r"[,\|;/]+", s)

    out = []
    for t in items:
        t = str(t).lower()
        t = t.replace("&", "and")
        t = re.sub(r"[\[\]\(\)\{\}\"']", " ", t) 
        t = re.sub(r"[^0-9a-z\s]+", " ", t) 
        t = re.sub(r"\s{2,}", " ", t).strip()
        if squash_spaces:
            t = t.replace(" ", "")
        if t:
            out.append(t)
    return set(out)

def _collect_terms_from_columns(row: pd.Series, columns: list[str], *, squash_spaces=False):
    terms = set()
    for col in columns:
        if col in row and pd.notna(row[col]):
            terms |= _norm_terms(row[col], squash_spaces=squash_spaces)
    return terms

def _collect_terms_dynamic(row: pd.Series, needles: list[str], *, squash_spaces=False):
    terms = set()
    for col in row.index:  # already lowercase (load_movies_df)
        if any(n in col for n in needles):
            val = row.get(col)
            if pd.notna(val):
                terms |= _norm_terms(val, squash_spaces=squash_spaces)
    return terms

def _collect_terms(row: pd.Series, static_cols: list[str], needles: list[str], *, squash_spaces=False):
    terms = _collect_terms_from_columns(row, static_cols, squash_spaces=squash_spaces)
    terms |= _collect_terms_dynamic(row, needles, squash_spaces=squash_spaces)
    return terms

def build_reason_from_rows(row_i: pd.Series, row_j: pd.Series, i: int = None, j: int = None):
    sim_mat = load_cosine_sim()
    score = None
    if sim_mat is not None and sim_mat.ndim == 2 and i is not None and j is not None:
        if i < sim_mat.shape[0] and j < sim_mat.shape[0]:
            score = float(sim_mat[i, j])
    if score is None:
        s_i = str(row_i.get("soup", ""))
        s_j = str(row_j.get("soup", ""))
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        X = tfidf.fit_transform([s_i, s_j])
        score = float(cosine_similarity(X)[0, 1])

    genres_i = _collect_terms(row_i, static_cols=["genres"], needles=["genre"], squash_spaces=True)
    genres_j = _collect_terms(row_j, static_cols=["genres"], needles=["genre"], squash_spaces=True)
    cast_i = _collect_terms(row_i, static_cols=["cast","actors","cast_names","actor_names"], needles=["cast","actor"])
    cast_j = _collect_terms(row_j, static_cols=["cast","actors","cast_names","actor_names"], needles=["cast","actor"])
    keys_i = _collect_terms(row_i, static_cols=["keywords","tags","tag","key_words"], needles=["keyword","tag"])
    keys_j = _collect_terms(row_j, static_cols=["keywords","tags","tag","key_words"], needles=["keyword","tag"])

    shared = {
        "genres":   sorted(list(genres_i & genres_j))[:10],
        "cast":     sorted(list(cast_i & cast_j))[:10],
        "keywords": sorted(list(keys_i & keys_j))[:12],
        "same_director": False,
        "director": None
    }

    di = str(row_i.get("director", "")).strip().lower()
    dj = str(row_j.get("director", "")).strip().lower()
    if di and dj and di == dj:
        shared["same_director"] = True
        shared["director"] = row_i.get("director")

    soup_i_top = dict(_tokenize_soup(row_i.get("soup", "")))
    soup_j_top = dict(_tokenize_soup(row_j.get("soup", "")))
    soup_overlap = sorted(((t, min(soup_i_top.get(t, 0), soup_j_top.get(t, 0)))
                           for t in set(soup_i_top) & set(soup_j_top)),
                          key=lambda x: (-x[1], x[0]))[:15]
    shared["soup_overlap"] = [t for t, _ in soup_overlap]

    year_i = _year(row_i.get("year", row_i.get("release_date", "")))
    year_j = _year(row_j.get("year", row_j.get("release_date", "")))

    return {
        "similarity": round(score, 4),
        "shared": shared,
        "year_i": year_i,
        "year_j": year_j,
        "year_gap": (abs(year_i - year_j) if (year_i and year_j) else None)
    }

_poster_cache: dict[tuple[str, int | None], str | None] = {}

def _normalize_title_for_search(title: str) -> str:
    t = str(title).strip()
    t = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t

def _soft_normalize(title: str) -> str:
    t = re.sub(r"[^\w\s]", " ", str(title))
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def tmdb_poster_for_title(title: str, year: int | None = None) -> str | None:
    if not title:
        return None

    key = os.environ.get("TMDB_API_KEY")
    if not key:
        app.logger.warning("TMDB_API_KEY missing; returning fallback")
        return None

    cache_key = (title.strip().lower(), year)
    if cache_key in _poster_cache:
        return _poster_cache[cache_key]

    attempts = [(title, year),(title, None),(_soft_normalize(title), None)]

    def _score(query_lower, r, y):
        title_r = (r.get("title") or r.get("name") or "").lower()
        s = 0
        if title_r == query_lower: s += 4
        if query_lower in title_r: s += 1
        rd = r.get("release_date") or ""
        try:
            ry = int(rd[:4]) if len(rd) >= 4 else None
        except Exception:
            ry = None
        if y and ry and ry == y: s += 2
        if r.get("poster_path"): s += 1
        pop = float(r.get("popularity") or 0.0)
        return (-s, -pop)

    base = "https://image.tmdb.org/t/p/" + os.environ.get("TMDB_IMAGE_SIZE", "w342")

    for q_title, q_year in attempts:
        q = _normalize_title_for_search(q_title)
        params = {"api_key": key, "query": q, "include_adult": "false"}
        if q_year:
            params["year"] = q_year

        try:
            resp = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=8)
            if resp.status_code != 200:
                app.logger.warning("TMDB %s (%s) -> %s", q, q_year, resp.status_code)
                continue

            data = resp.json() or {}
            results = data.get("results") or []
            if not results:
                continue

            query_lower = q.lower()
            results.sort(key=lambda r: _score(query_lower, r, q_year))

            for r in results:
                pp = r.get("poster_path")
                if pp:
                    url = f"{base}{pp}"
                    _poster_cache[cache_key] = url
                    return url
            continue

        except Exception as e:
            app.logger.warning("TMDB request error for %r (%s): %s", q, q_year, e)
            continue

    _poster_cache[cache_key] = None
    return None

def poster_from_row_or_tmdb(row: pd.Series, fallback_title: str | None = None) -> tuple[str | None, int | None]:
    title = str(row.get("title")) if row is not None and "title" in row else (fallback_title or "")
    year  = _year(row.get("year", row.get("release_date", ""))) if row is not None else None
    url = tmdb_poster_for_title(title, year)
    return url, year

def poster_url_for(title: str, year: int | None):
    sig = hashlib.md5(f"{title}|{year or ''}".encode("utf-8")).hexdigest()[:8]
    return url_for("poster_jpg", title=title, year=year, v=sig)

@app.route("/poster.jpg")
def poster_jpg():
    title = request.args.get("title", "")
    year  = request.args.get("year", type=int)
    _ = request.args.get("v") 

    url = tmdb_poster_for_title(title, year)
    if not url:
        return Response("Poster not found", status=404, mimetype="text/plain", headers={"Cache-Control": "no-store, max-age=0"})

    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Encoding": "identity",
        }
        r = requests.get(url, headers=headers, stream=True, timeout=15)
        r.raise_for_status()

        def generate():
            for chunk in r.iter_content(65536):
                if chunk: 
                    yield chunk

        resp = Response(generate(), mimetype=r.headers.get("Content-Type", "image/jpeg"))
        if "Content-Length" in r.headers:
            resp.headers["Content-Length"] = r.headers["Content-Length"]
        resp.headers["Cache-Control"] = "public, max-age=604800"
        return resp
    except Exception:
        return Response("Poster fetch error", status=404, mimetype="text/plain", headers={"Cache-Control": "no-store, max-age=0"})


recommender = MovieRecommender(csv_path=str(CSV_PATH))
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        title = (request.form.get("movie_title")
                 or (request.json or {}).get("movie_title")
                 or "").strip()

        if not title:
            return jsonify({"error": "movie_title missing", "recommendations": []}), 400

        df = load_movies_df()

        rec_df = recommender.get_recommendations(title, k=25)
        if isinstance(rec_df, pd.DataFrame):
            if "score" in rec_df.columns:
                rec_df = rec_df.sort_values("score", ascending=False)
            base_records = rec_df.fillna("").to_dict("records")
        elif isinstance(rec_df, list):
            base_records = [{"title": str(x)} for x in rec_df]
        else:
            base_records = []

        i = title_to_index(title)
        row_i = df.iloc[i] if i is not None else None

        input_year = _year(row_i.get("year", row_i.get("release_date", ""))) if row_i is not None else None
        input_poster = poster_url_for(title, input_year)

        enriched = []
        for rec in base_records:
            rec_title = str(rec.get("title", "")).strip()
            if not rec_title:
                continue
            j = title_to_index(rec_title)
            row_j = df.iloc[j] if j is not None else None

            why = None
            try:
                if row_i is not None and row_j is not None:
                    why = build_reason_from_rows(row_i, row_j, i, j)
                elif row_j is not None:
                    dummy_src = pd.Series({"soup": row_i.get("soup", "") if row_i is not None else title})
                    why = build_reason_from_rows(dummy_src, row_j, None, None)
            except Exception as e:
                app.logger.warning("why-build failed for %r: %s", rec_title, e)

            rec_year  = _year(row_j.get("year", row_j.get("release_date", ""))) if row_j is not None else None
            poster_so = poster_url_for(rec_title, rec_year)

            extra = {k: v for k, v in rec.items() if k not in {"title"}}
            enriched.append({
                "title": rec_title,
                "poster": poster_so,
                "why": why,
                **extra
            })

        for r in enriched:
            sim = (r.get("why") or {}).get("similarity")
            try:
                r["_score"] = float(sim) if sim is not None else float("-inf")
            except Exception:
                r["_score"] = float("-inf")

        def _rank_key(r):
            why = r.get("why") or {}
            shared = (why.get("shared") or {})
            bonus = 0.05 if shared.get("same_director") else 0.0
            yg = why.get("year_gap")
            penalty = 0.0 if yg is None else (min(yg, 50) / 1000.0)
            return r["_score"] + bonus - penalty

        seen = set()
        src_lower = title.strip().lower()
        filtered = []
        for r in enriched:
            t = (r.get("title") or "").strip()
            if not t or t.lower() == src_lower:
                continue
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            filtered.append(r)

        filtered.sort(key=_rank_key, reverse=True)

        K = int(request.args.get("k") or 10)
        enriched = filtered[:K]

        for r in enriched:
            r.pop("_score", None)

        app.logger.info("recommend (re-ranked by cosine): '%s' -> %d recs", title, len(enriched))

        return jsonify({
            "input": title,
            "input_poster": input_poster,
            "input_year": input_year,
            "recommendations": enriched
        })

    except Exception as e:
        app.logger.exception("recommend failed: %s", e)
        return jsonify({"error": "internal error", "recommendations": []}), 500    

@app.route("/stats.json", methods=["GET"])
def stats_json():
    df = load_movies_df()
    totals = {
        "rows": int(len(df)),
        "unique_titles": int(df["title"].nunique()) if "title" in df.columns else None,
        "unique_directors": int(df["director"].nunique()) if "director" in df.columns else None,
    }
    missing = df.isnull().sum().to_dict()

    def hist(series, bins=20):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty: return None
        counts, edges = np.histogram(s, bins=bins)
        return {"counts": counts.tolist(), "edges": edges.tolist()}

    top_genres = []
    if "genres" in df.columns:
        split_genres = df["genres"].dropna().astype(str).str.split(",").explode().str.strip()
        genre_counts = split_genres.value_counts().head(15)
        top_genres = [{"genre": g, "count": int(c)} for g, c in genre_counts.items()]

    df["title_len"] = df["title"].fillna("").str.len() if "title" in df.columns else 0
    df["soup_len"]  = df["soup"].fillna("").str.len()  if "soup"  in df.columns else 0

    rating_hist     = hist(df["rating"])     if "rating"     in df.columns else None
    popularity_hist = hist(df["popularity"]) if "popularity" in df.columns else None
    soup_len_hist   = hist(df["soup_len"])
    title_len_hist  = hist(df["title_len"])

    sim = load_cosine_sim()
    if sim is not None and sim.ndim == 2:
        n = min(30, sim.shape[0])
        sim_sample = sim[:n, :n].round(4).tolist()
        n_titles = n
    else:
        n_titles = min(30, len(df))
        sim_sample = sample_similarity_from_soup(df, n=n_titles)

    sample_titles = df["title"].fillna("").astype(str).head(n_titles).tolist()

    return jsonify({
        "totals": totals,
        "missing": missing,
        "top_genres": top_genres,
        "has_rating": rating_hist is not None,
        "has_popularity": popularity_hist is not None,
        "rating_hist": rating_hist,
        "popularity_hist": popularity_hist,
        "soup_len_hist": soup_len_hist,
        "title_len_hist": title_len_hist,
        "similarity_sample": sim_sample,
        "sample_titles": sample_titles
    })

@app.route("/reason.json", methods=["GET"])
def reason_json():
    df = load_movies_df()
    sim = load_cosine_sim()
    n = min(30, sim.shape[0]) if sim is not None and sim.ndim == 2 else min(30, len(df))
    try:
        i = int(request.args.get("i", 0))
        j = int(request.args.get("j", 0))
    except Exception:
        return jsonify({"error": "invalid indices"}), 400
    if i < 0 or j < 0 or i >= n or j >= n:
        return jsonify({"error": f"indices out of range [0, {n-1}]"}), 400
    row_i = df.iloc[i]
    row_j = df.iloc[j]
    reason = build_reason_from_rows(row_i, row_j, i, j)
    return jsonify({ "i": i, "j": j, "title_i": str(row_i.get("title", f"Row {i}")),
                     "title_j": str(row_j.get("title", f"Row {j}")), **reason })
    
def _fig_bytes(fig, dpi=160):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _primary_genre(s):
    if pd.isna(s): return None
    parts = [g.strip().lower() for g in str(s).split(",") if g.strip()]
    return parts[0] if parts else None

# Genre Confusion Matrix Visualization
@app.route("/viz/confusion.png")
def confusion_png():
    df = load_movies_df().copy()

    top = request.args.get("top", type=int) or 12
    normalize = bool(request.args.get("normalize", 1))
    fs = request.args.get("fs", type=float) or 9.0
    vfs = request.args.get("vfs", type=float) or 7.0
    max_feats = request.args.get("maxf", type=int) or 20000

    if "genres" not in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No 'genres' column", ha="center", va="center")
        ax.axis("off")
        png = _fig_bytes(fig)
        return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store, max-age=0"})

    df["primary_genre"] = df["genres"].map(_primary_genre)
    df = df.dropna(subset=["primary_genre"])

    top_genres = df["primary_genre"].value_counts().head(top).index.tolist()
    df = df[df["primary_genre"].isin(top_genres)]
    if len(df) < 100 or len(top_genres) < 2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Not enough genre variety/rows", ha="center", va="center")
        ax.axis("off")
        png = _fig_bytes(fig)
        return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store, max-age=0"})

    text_col = "overview" if "overview" in df.columns else ("soup" if "soup" in df.columns else "title")
    X_text = df[text_col].fillna("").astype(str)
    y = df["primary_genre"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.25, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2),
                          min_df=2, max_features=max_feats, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=300, solver="liblinear", multi_class="ovr")
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)

    labels = top_genres
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        M = cm_norm
        fmt_value = lambda v: f"{v:.2f}"
        title = "Genre Confusion (row-normalized)"
    else:
        M = cm.astype(float)
        fmt_value = lambda v: f"{int(v)}"
        title = "Genre Confusion (counts)"

    plt.style.use("dark_background")
    side = min(28, max(12, 0.8 * len(labels) + 4))
    fig, ax = plt.subplots(figsize=(side, side), constrained_layout=True)

    im = ax.imshow(M, cmap="magma")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion" if normalize else "Count", rotation=270, labelpad=15)

    ax.set_title(title)
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            color = "white" if (normalize and v >= 0.5) or (not normalize and v >= M.max()*0.6) else "#e5e7eb"
            txt = ax.text(j, i, fmt_value(v), ha="center", va="center", fontsize=vfs, color=color)
            txt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="black", alpha=0.6)])

    ax.grid(False)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Response(buf.getvalue(), mimetype="image/png",
                    headers={"Cache-Control": "no-store, max-age=0"})

# Distribution Visualization
@app.route("/viz/dist.png")
def dist_png():
    df = load_movies_df().copy()

    if "title" in df.columns:
        df["title_len"] = df["title"].fillna("").astype(str).str.len()
    if "soup" in df.columns:
        df["soup_len"] = df["soup"].fillna("").astype(str).str.len()

    if "year" not in df.columns and "release_date" in df.columns:
        y = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        if y.notna().sum() > 0:
            df["year"] = y

    user_metric = request.args.get("metric")
    candidates = ([user_metric] if user_metric else []) + [
        "rating", "vote_average",
        "popularity", "vote_count",
        "runtime", "year",
        "soup_len", "title_len",
    ]

    def pick_series(cols):
        for col in cols:
            if not col or col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.size >= 10 and s.var() > 0:
                return col, s
        return None, None

    chosen, series = pick_series(candidates)
    if series is None:
        for force in ("soup_len", "title_len"):
            if force in df.columns:
                s = pd.to_numeric(df[force], errors="coerce").dropna()
                if s.size > 0:
                    chosen, series = force, s
                    break

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))

    if series is None or series.empty:
        ax.text(0.5, 0.5, "No numeric data available", ha="center", va="center")
        ax.axis("off")
    else:
        ax.hist(series.values, bins=25)
        label = chosen.replace("_", " ").title()
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Distribution")
        ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Response(buf.getvalue(), mimetype="image/png", headers={"Cache-Control": "no-store, max-age=0"})

# Heatmap Visualization
@app.route("/viz/heatmap.png")
def heatmap_png():
    df = load_movies_df()
    sim = load_cosine_sim()

    n_req  = request.args.get("n", type=int) or 30
    rotate = request.args.get("rotate", default="90")
    try:
        rotate = float(rotate)
    except Exception:
        rotate = 90.0
    wrap = request.args.get("wrap", type=int) or 0
    fs = request.args.get("fs", type=float) or 8.0
    vfs = request.args.get("vfs", type=float) or 6.0

    if sim is not None and sim.ndim == 2:
        n = min(n_req, sim.shape[0])
        M = sim[:n, :n]
    else:
        n = min(n_req, len(df))
        M = np.array(sample_similarity_from_soup(df, n=n)) if n > 1 else np.zeros((1, 1))

    titles = df["title"].fillna("").astype(str).head(n).tolist()

    if wrap and wrap > 0:
        import textwrap
        def wrap_label(s): 
            out = "\n".join(textwrap.wrap(s, width=wrap))
            return out if out else s
        xlabels = [wrap_label(t) for t in titles]
        ylabels = [wrap_label(t) for t in titles]
    else:
        xlabels = titles
        ylabels = titles

    plt.style.use("dark_background")

    side = min(30, max(12, 0.35 * n + 6))
    fig, ax = plt.subplots(figsize=(side, side), constrained_layout=True)

    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="magma", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", rotation=270, labelpad=15)

    ax.set_title("Similarity Heatmap (sample)")
    ax.set_xlabel("Movies")
    ax.set_ylabel("Movies")

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(xlabels, rotation=rotate, fontsize=fs)
    ax.set_yticklabels(ylabels, fontsize=fs)

    for i in range(n):
        for j in range(n):
            v = float(M[i, j])
            color = "white" if v >= 0.6 else "#d1d5db"
            txt = ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=vfs, color=color)
            txt.set_path_effects([pe.withStroke(linewidth=1.0, foreground="black", alpha=0.6)])

    ax.grid(False)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Response(buf.getvalue(), mimetype="image/png",
                    headers={"Cache-Control": "no-store, max-age=0"})

@app.after_request
def no_cache(resp):
    if not request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)