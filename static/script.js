(() => {
  const escapeHtml = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

  function chips(arr){
    return (arr && arr.length)
      ? arr.map(t => `<span class="chip">${escapeHtml(t)}</span>`).join('')
      : '<em class="muted">None</em>';
  }

  function replaceWithPosterError(imgEl, msg = "Poster unavailable") {
    if (!imgEl) return;
    const err = document.createElement("div");
    err.className = "poster-error";
    err.textContent = msg;
    imgEl.replaceWith(err);
  }

  function openReasonModal(innerHtml, titleText){
    const modal = document.getElementById("reasonModal");
    if (!modal) return;
    document.getElementById("reasonBody").innerHTML = innerHtml || '';
    document.getElementById("reasonTitle").textContent = titleText || "Why this?";
    modal.style.display = "block";
  }
  function closeReasonModal(){
    const modal = document.getElementById("reasonModal");
    if (!modal) return;
    modal.style.display = "none";
  }
  (function wireModal(){
    const m = document.getElementById("reasonModal");
    if (!m) return;
    const btn = document.getElementById("reasonClose");
    btn && (btn.onclick = closeReasonModal);
    m.addEventListener("click", (e) => { if (e.target.id === "reasonModal") closeReasonModal(); });
    window.addEventListener("keydown", (e) => { if (e.key === "Escape") closeReasonModal(); });
  })();

  function initHome(){
    const form = document.getElementById('recommendForm');
    if (!form) return;

    const sourceEl = document.getElementById('source');
    const sourceTitleEl = document.getElementById('sourceTitle');
    const recsSection = document.getElementById('recsSection');
    const recsGrid = document.getElementById('recs');

    let LAST_INPUT_TITLE = '';
    let LAST_RECS = [];

    if (!recsGrid.dataset.wired) {
      recsGrid.addEventListener('click', (e) => {
        const tile = e.target.closest('.tile');
        if (!tile) return;
        const idx = Number(tile.dataset.index || -1);
        if (!(idx >= 0 && idx < LAST_RECS.length)) return;

        const rec = LAST_RECS[idx];
        const why = rec.why;
        const a = LAST_INPUT_TITLE || 'Input';
        const b = rec.title || 'Recommendation';

        const sim = (why && why.similarity != null && !isNaN(why.similarity)) ? Number(why.similarity).toFixed(3) : '';
        const years = (why && why.year_i && why.year_j) ? ` • Year gap: ${why.year_gap}` : '';
        const body = why ? `
          <div style="margin-bottom:8px;">
            <div><strong>${escapeHtml(a)}</strong> ↔ <strong>${escapeHtml(b)}</strong></div>
            <div class="muted">Similarity: <strong>${sim}</strong>${years}</div>
          </div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
            <div>
              <div class="muted" style="margin-bottom:4px;">Shared genres</div>
              ${chips(why.shared?.genres)}
              <div class="muted" style="margin:10px 0 4px;">Shared cast</div>
              ${chips(why.shared?.cast)}
              <div class="muted" style="margin:10px 0 4px;">Shared keywords</div>
              ${chips(why.shared?.keywords)}
            </div>
            <div>
              <div class="muted" style="margin-bottom:4px;">Overlap in “soup”</div>
              ${chips(why.shared?.soup_overlap)}
              <div class="muted" style="margin:10px 0 4px;">Same director?</div>
              <div>${why.shared?.same_director ? `Yes — <strong>${escapeHtml(why.shared?.director || '')}</strong>` : 'No / N/A'}</div>
            </div>
          </div>` : `<em class="muted">Sorry, we couldn't find any similar movies for the movie provided.</em>`;
        openReasonModal(body, 'Why this recommendation?');
      });
      recsGrid.dataset.wired = "1";
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      recsGrid.innerHTML = '<div class="muted">Loading…</div>';

      const fd = new FormData(form);
      let data;
      try {
        const res = await fetch('/recommend', { method: 'POST', body: fd });
        if (!res.ok) throw new Error('Bad response');
        data = await res.json();
      } catch (err) {
        recsGrid.innerHTML = '<div class="muted">Failed to get recommendations.</div>';
        return;
      }

      LAST_INPUT_TITLE = data.input || '';
      LAST_RECS = data.recommendations || [];

      let posterEl = document.getElementById('sourcePoster');
      if (!posterEl || posterEl.tagName !== 'IMG') {
        const old = document.getElementById('sourcePoster');
        if (old && old.parentElement) old.parentElement.removeChild(old);
        posterEl = document.createElement('img');
        posterEl.id = 'sourcePoster';
        posterEl.className = 'poster';
        if (sourceEl.firstChild) sourceEl.insertBefore(posterEl, sourceEl.firstChild);
        else sourceEl.appendChild(posterEl);
      }

      posterEl.alt = data.input || 'Poster';
      posterEl.title = data.input || '';
      if (data.input_poster) {
        posterEl.onerror = () => replaceWithPosterError(posterEl, "Poster unavailable");
        posterEl.src = data.input_poster;  // set AFTER onerror
      } else {
        replaceWithPosterError(posterEl, "Poster unavailable");
      }

      if (sourceTitleEl) sourceTitleEl.textContent = data.input || '';
      if (sourceEl) sourceEl.style.display = 'flex';

      recsGrid.innerHTML = '';
      if (!LAST_RECS.length){
        recsGrid.innerHTML = '<em class="muted">No results.</em>';
        recsSection && (recsSection.style.display = 'block');
        return;
      }

      LAST_RECS.forEach((rec, idx) => {
        const tile = document.createElement('div');
        tile.className = 'tile';
        tile.dataset.index = String(idx);

        const img = document.createElement('img');
        img.className = 'poster';
        img.alt = rec.title || 'Poster';
        img.title = rec.title || '';
        if (rec.poster) {
          img.onerror = () => replaceWithPosterError(img, `No poster for “${rec.title || "this movie"}”`);
          img.src = rec.poster; // AFTER onerror
        } else {
          replaceWithPosterError(img, `No poster for “${rec.title || "this movie"}”`);
        }

        const cap = document.createElement('div');
        cap.className = 'caption';
        cap.textContent = rec.title || '';

        tile.appendChild(img);
        tile.appendChild(cap);
        recsGrid.appendChild(tile);
      });
      recsSection && (recsSection.style.display = 'block');
    });
  }

  document.addEventListener('DOMContentLoaded', function() {
    const recommendBtn = document.getElementById('recommend-btn');
    const movieInput = document.getElementById('movie-input');
    const recommendationsList = document.getElementById('recommendations-list');
    const inputTitle = document.getElementById('input-title');
    const inputMovieContainer = document.querySelector('.input-movie');

    if (!recommendBtn || !movieInput || !recommendationsList) {
      if (document.getElementById('recommendForm')) initHome();
      return;
    }

    recommendBtn.addEventListener('click', getRecommendations);
    movieInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') getRecommendations();
    });

    function getRecommendations() {
      const title = movieInput.value.trim();
      if (!title) return;

      recommendationsList.innerHTML = '<div class="loading">Finding recommendations...</div>';

      fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `movie_title=${encodeURIComponent(title)}`
      })
      .then(r => r.json())
      .then(data => displayRecommendations(title, data))
      .catch(err => {
        recommendationsList.innerHTML = '<div class="error">Error fetching recommendations. Please try again.</div>';
        console.error(err);
      });
    }

    function displayRecommendations(inputTitleText, data) {
      recommendationsList.innerHTML = '';

      if (inputMovieContainer) {
        inputMovieContainer.style.display = 'block';
        inputTitle.textContent = data.input || inputTitleText;

        let poster = inputMovieContainer.querySelector('img.poster');
        if (!poster) {
          poster = document.createElement('img');
          poster.className = 'poster';
          inputMovieContainer.prepend(poster);
        }

        poster.alt = data.input || 'Poster';
        poster.title = data.input || '';
        if (data.input_poster) {
          poster.onerror = () => replaceWithPosterError(poster, "Poster unavailable");
          poster.src = data.input_poster;
        } else {
          replaceWithPosterError(poster, "Poster unavailable");
        }
      }

      if (!data || !Array.isArray(data.recommendations) || data.recommendations.length === 0) {
        recommendationsList.innerHTML = '<div class="error">No recommendations found. Try another movie.</div>';
        return;
      }

      data.recommendations.forEach((rec, idx) => {
        const card = document.createElement('div');
        card.className = 'movie-card';
        card.dataset.index = String(idx);

        const img = document.createElement('img');
        img.className = 'poster';
        img.alt = rec.title || 'Poster';
        img.title = rec.title || '';
        if (rec.poster) {
          img.onerror = () => replaceWithPosterError(img, `No poster for “${rec.title || "this movie"}”`);
          img.src = rec.poster;
        } else {
          replaceWithPosterError(img, `No poster for “${rec.title || "this movie"}”`);
        }

        const titleDiv = document.createElement('div');
        titleDiv.className = 'movie-title';
        titleDiv.textContent = rec.title || '';

        const meta = document.createElement('div');
        meta.className = 'movie-meta';
        meta.innerHTML = [
          rec.genres ? 'Genres: ' + escapeHtml(rec.genres) : '',
          rec.director ? 'Director: ' + escapeHtml(rec.director) : ''
        ].filter(Boolean).join('<br>');

        card.appendChild(img);
        card.appendChild(titleDiv);
        if (meta.innerHTML) card.appendChild(meta);

        card.addEventListener('click', () => {
          const why = rec.why;
          const a = data.input || inputTitleText || 'Input';
          const b = rec.title || 'Recommendation';
          const sim = (why && why.similarity != null && !isNaN(why.similarity)) ? Number(why.similarity).toFixed(3) : '';
          const years = (why && why.year_i && why.year_j) ? ` • Year gap: ${why.year_gap}` : '';
          const body = why ? `
            <div style="margin-bottom:8px;">
              <div><strong>${escapeHtml(a)}</strong> ↔ <strong>${escapeHtml(b)}</strong></div>
              <div class="muted">Similarity: <strong>${sim}</strong>${years}</div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
              <div>
                <div class="muted" style="margin-bottom:4px;">Shared genres</div>
                ${chips(why.shared?.genres)}
                <div class="muted" style="margin:10px 0 4px;">Shared cast</div>
                ${chips(why.shared?.cast)}
                <div class="muted" style="margin:10px 0 4px;">Shared keywords</div>
                ${chips(why.shared?.keywords)}
              </div>
              <div>
                <div class="muted" style="margin-bottom:4px;">Overlap in “soup”</div>
                ${chips(why.shared?.soup_overlap)}
                <div class="muted" style="margin:10px 0 4px;">Same director?</div>
                <div>${why.shared?.same_director ? `Yes — <strong>${escapeHtml(why.shared?.director || '')}</strong>` : 'No / N/A'}</div>
              </div>
            </div>` : `<em class="muted">Sorry, we couldn't find any similar movies for the movie provided.</em>`;
          openReasonModal(body, 'Why this recommendation?');
        });

        recommendationsList.appendChild(card);
      });
    }
  });

  if (document.getElementById('recommendForm')) initHome();
})();