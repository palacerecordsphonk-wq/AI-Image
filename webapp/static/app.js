// ----- Helpers ---------------------------------------------------------------
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

async function api(method, url, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const res = await fetch(url, opt);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

let toastTimer;
function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (isErr ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), 3500);
}

// ----- Tabs ------------------------------------------------------------------
$$(".tab").forEach((tab) => tab.addEventListener("click", () => {
  $$(".tab").forEach((t) => t.classList.remove("active"));
  $$(".view").forEach((v) => v.classList.remove("active"));
  tab.classList.add("active");
  $("#view-" + tab.dataset.view).classList.add("active");
  if (tab.dataset.view === "generate") loadStyleOptions();
  if (tab.dataset.view === "spotify") loadSpotifyStyles();
  if (tab.dataset.view === "gallery") loadGallery();
}));

// ----- Job polling -----------------------------------------------------------
let pollTimer = null;
function setBadge(on, text) {
  const b = $("#jobBadge");
  b.classList.toggle("show", on);
  if (text) $("#jobBadgeText").textContent = text;
}

function pollJob(id, { logEl, statusEl, onDone }) {
  clearInterval(pollTimer);
  const tick = async () => {
    let j;
    try { j = await api("GET", "/api/jobs/" + id); } catch (e) { return; }
    if (logEl) {
      const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 30;
      logEl.style.display = "block";
      logEl.textContent = j.log || "";
      if (atBottom) logEl.scrollTop = logEl.scrollHeight;
    }
    if (statusEl) statusEl.innerHTML = statusPill(j);
    setBadge(j.status === "running", `${j.kind} : ${j.style || ""}`);
    if (j.status !== "running") {
      clearInterval(pollTimer);
      setBadge(false);
      if (onDone) onDone(j);
    }
  };
  tick();
  pollTimer = setInterval(tick, 1500);
}

function statusPill(j) {
  const map = { running: "en cours…", done: "terminé ✓", error: "erreur", stopped: "arrêté" };
  let html = `<div class="status-line"><span class="pill ${j.status}">${map[j.status] || j.status}</span>`;
  if (j.status === "running") html += `<button class="ghost danger" onclick="stopJob('${j.id}')">Arrêter</button>`;
  html += `</div>`;
  return html;
}

async function stopJob(id) { await api("POST", `/api/jobs/${id}/stop`); toast("Tâche arrêtée."); }

// ----- Styles list -----------------------------------------------------------
let STYLES = [];
let currentStyle = null;

async function loadStyles() {
  STYLES = await api("GET", "/api/styles");
  const list = $("#styleList");
  if (!STYLES.length) {
    list.innerHTML = `<div class="empty" style="padding:20px">Aucun style. Crée le premier ↓</div>`;
    return;
  }
  list.innerHTML = STYLES.map((s) => `
    <div class="style-item ${s.name === currentStyle ? "active" : ""}" data-name="${s.name}">
      <div class="row">
        <span class="nm">${s.name}</span>
        ${s.trained ? '<span class="tag ok">entraîné</span>' : '<span class="tag">non entraîné</span>'}
      </div>
      <div class="style-meta">${s.raw_count} img · trigger: ${s.trigger}</div>
    </div>`).join("");
  $$(".style-item", list).forEach((el) =>
    el.addEventListener("click", () => selectStyle(el.dataset.name)));
}

function selectStyle(name) {
  currentStyle = name;
  loadStyles();
  renderDetail();
}

async function renderDetail() {
  const s = STYLES.find((x) => x.name === currentStyle);
  if (!s) return;
  const wrap = $("#styleDetail");
  wrap.innerHTML = `
    <div class="card">
      <div class="step-title"><h2>${s.name}</h2>
        ${s.trained ? '<span class="tag ok">entraîné</span>' : '<span class="tag">non entraîné</span>'}</div>
      <h3>trigger : <b>${s.trigger}</b> · titre : ${s.title_text} · ${s.raw_count} images importées</h3>
    </div>

    <div class="card">
      <div class="step-title"><span class="step-num">1</span><h2>Ajouter des images</h2></div>
      <label>Chemin d'un dossier sur ton Mac (le plus simple)</label>
      <input type="text" id="impPath" placeholder="/Users/toi/Desktop/rock" />
      <div class="btn-row">
        <button id="btnImportPath">Importer depuis le dossier</button>
        <button class="ghost" id="btnUpload">…ou choisir des fichiers</button>
        <input type="file" id="fileInput" multiple accept="image/*" style="display:none" />
      </div>
      <div class="hint">Tu peux relancer plusieurs fois pour ajouter d'autres images.</div>
      <div class="thumbs" id="thumbs"></div>
    </div>

    <div class="card">
      <div class="step-title"><span class="step-num">2</span><h2>Préparer & légender</h2></div>
      <div class="row-inputs">
        <div><label>Résolution des images</label>
          <select id="prepSize">
            <option value="768">768 (rapide)</option>
            <option value="1024" selected>1024 (qualité)</option>
            <option value="1536">1536 (haute)</option>
            <option value="2048">2048 (très haute)</option>
            <option value="3000">3000 (ultra)</option>
          </select>
        </div>
        <div><label>Méthode de captioning (IA)</label>
          <select id="prepBackend">
            <option value="gemini" selected>Gemini (API — meilleur OCR du titre)</option>
            <option value="local">Modèle local (~5 Go, hors-ligne, Mac)</option>
            <option value="simple">Aucune (légende minimale)</option>
          </select>
        </div>
      </div>

      <div id="prepGeminiBox" class="key-box">
        <label>🔑 Clé API Gemini</label>
        <input type="password" id="prepGeminiKey" placeholder="Colle ta clé Gemini (AIza…)" />
        <div class="hint">Gratuite sur <b>aistudio.google.com/apikey</b>. Mémorisée uniquement dans ce navigateur (localStorage), envoyée seulement à l'API Gemini pour décrire tes covers.</div>
      </div>

      <div class="hint">Résolution = définition à laquelle les covers sont stockées (Ultra = pleine qualité HD). L'auto-captioning décrit chaque image (couleurs, ambiance, perso, typo) et détecte le titre. <b>Gemini</b> lit bien mieux les titres stylisés/cachés ; le <b>modèle local</b> reste hors-ligne et gratuit (télécharge ~5 Go au 1er usage).</div>
      <div class="btn-row"><button id="btnPrepare">Préparer le dataset</button></div>
    </div>

    <div class="card">
      <div class="step-title"><span class="step-num">3</span><h2>Entraîner le style (LoRA)</h2></div>
      <div class="row-inputs three">
        <div><label>Résolution d'entraînement</label>
          <select id="trMaxRes">
            <option value="768">768 (rapide)</option>
            <option value="1024" selected>1024 (recommandé)</option>
            <option value="1536">1536 (lent)</option>
          </select></div>
        <div><label>Itérations (~)</label><input type="number" id="trSteps" value="1200" /></div>
        <div><label>Rang LoRA</label><input type="number" id="trRank" value="16" /></div>
      </div>
      <div class="hint">⚠️ Ici c'est la résolution d'<b>entraînement</b> (≠ stockage). FLUX est optimisé pour ~1024 : au-delà c'est beaucoup plus lent sans vrai gain. Pour des covers ultra-nettes, génère puis upscale.</div>
      <div class="btn-row"><button id="btnTrain">🚀 Lancer l'entraînement</button></div>
    </div>

    <div class="card">
      <div class="step-title"><h2>Journal</h2></div>
      <div id="jobStatus"></div>
      <div class="log" id="jobLog" style="display:none"></div>
    </div>`;

  loadThumbs();
  $("#btnImportPath").onclick = importPath;
  $("#btnUpload").onclick = () => $("#fileInput").click();
  $("#fileInput").onchange = uploadFiles;
  $("#btnPrepare").onclick = startPrepare;
  $("#btnTrain").onclick = startTrain;

  // Captioning : restaure la clé Gemini mémorisée + montre la zone selon la méthode.
  $("#prepGeminiKey").value = localStorage.getItem("gemini_key") || "";
  const syncBackend = () => {
    $("#prepGeminiBox").style.display = $("#prepBackend").value === "gemini" ? "block" : "none";
  };
  $("#prepBackend").onchange = syncBackend;
  syncBackend();
}

async function loadThumbs() {
  try {
    const data = await api("GET", `/api/styles/${currentStyle}/images`);
    const el = $("#thumbs");
    if (!el) return;
    el.innerHTML = data.images.map((u) => `<img src="${u}" loading="lazy" />`).join("");
  } catch (e) {}
}

async function importPath() {
  const path = $("#impPath").value.trim();
  if (!path) return toast("Indique un chemin de dossier.", true);
  try {
    const r = await api("POST", `/api/styles/${currentStyle}/import-path`, { path });
    toast(`${r.imported} image(s) importée(s).`);
    await loadStyles(); selectStyle(currentStyle);
  } catch (e) { toast(e.message, true); }
}

async function uploadFiles(ev) {
  const files = ev.target.files;
  if (!files.length) return;
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  try {
    const res = await fetch(`/api/styles/${currentStyle}/upload`, { method: "POST", body: fd });
    const r = await res.json();
    toast(`${r.imported} image(s) ajoutée(s).`);
    await loadStyles(); selectStyle(currentStyle);
  } catch (e) { toast("Échec de l'upload.", true); }
}

async function startPrepare() {
  const backend = $("#prepBackend").value;
  const geminiKey = $("#prepGeminiKey").value.trim();
  if (backend === "gemini") {
    if (!geminiKey) return toast("Colle ta clé API Gemini (ou choisis une autre méthode).", true);
    localStorage.setItem("gemini_key", geminiKey);  // mémorise dans ce navigateur
  }
  try {
    const job = await api("POST", `/api/styles/${currentStyle}/prepare`, {
      size: +$("#prepSize").value,
      caption_backend: backend,
      gemini_api_key: backend === "gemini" ? geminiKey : null,
    });
    pollJob(job.id, {
      logEl: $("#jobLog"), statusEl: $("#jobStatus"),
      onDone: () => { loadThumbs(); loadStyles(); toast("Préparation terminée."); },
    });
  } catch (e) { toast(e.message, true); }
}

async function startTrain() {
  try {
    const job = await api("POST", `/api/styles/${currentStyle}/train`, {
      max_resolution: +$("#trMaxRes").value, total_steps: +$("#trSteps").value, rank: +$("#trRank").value,
    });
    toast("Entraînement lancé. Tu peux suivre le journal.");
    pollJob(job.id, {
      logEl: $("#jobLog"), statusEl: $("#jobStatus"),
      onDone: (j) => { loadStyles(); toast(j.status === "done" ? "Entraînement terminé ✓" : "Entraînement : " + j.status, j.status !== "done"); },
    });
  } catch (e) { toast(e.message, true); }
}

// ----- New style modal -------------------------------------------------------
$("#btnNewStyle").onclick = () => $("#modalNew").classList.add("show");
$("#nsCancel").onclick = () => $("#modalNew").classList.remove("show");
$("#nsCreate").onclick = async () => {
  const name = $("#nsName").value.trim();
  if (!name) return toast("Donne un nom.", true);
  try {
    await api("POST", "/api/styles", {
      name, class_word: $("#nsClass").value.trim() || null, title_text: $("#nsTitle").value,
    });
    $("#modalNew").classList.remove("show");
    $("#nsName").value = ""; $("#nsClass").value = "";
    await loadStyles(); selectStyle(name);
    toast("Style créé.");
  } catch (e) { toast(e.message, true); }
};

// ----- Generation ------------------------------------------------------------
async function loadStyleOptions() {
  const styles = await api("GET", "/api/styles");
  const sel = $("#genStyle");
  sel.innerHTML = styles.map((s) =>
    `<option value="${s.name}">${s.name}${s.trained ? "" : " (non entraîné)"}</option>`).join("");
}

async function runGenerate(random = false) {
  const body = {
    style: $("#genStyle").value,
    title: $("#genTitle").value,
    extra: $("#genExtra").value,
    title_mode: $("#genTitleMode").value,
    grain: $("#genGrain").value,
    random,
    width: +$("#genW").value, height: +$("#genH").value,
    steps: +$("#genSteps").value, guidance: +$("#genGuid").value,
    lora_scale: +$("#genScale").value,
    // En aléatoire on force une seed aléatoire (champ ignoré) pour varier à chaque clic.
    seed: random ? null : ($("#genSeed").value ? +$("#genSeed").value : null),
  };
  if (!body.style) return toast("Choisis un style.", true);
  try {
    const job = await api("POST", "/api/generate", body);
    $("#btnGenerate").disabled = true;
    $("#btnRandom").disabled = true;
    $("#genPreview").innerHTML = `<div class="placeholder">${random ? "🎲 Inspiration en cours…" : "Génération en cours…"}</div>`;
    pollJob(job.id, {
      logEl: $("#genLog"), statusEl: $("#genStatus"),
      onDone: (j) => {
        $("#btnGenerate").disabled = false;
        $("#btnRandom").disabled = false;
        if (j.status === "done" && j.result && j.result.image) {
          $("#genPreview").innerHTML = `<img src="${j.result.image}?t=${Date.now()}" />`;
          toast("Image générée ✓");
        } else {
          $("#genPreview").innerHTML = `<div class="placeholder">Échec de la génération. Vois le journal.</div>`;
          toast("La génération a échoué.", true);
        }
      },
    });
  } catch (e) { toast(e.message, true); $("#btnGenerate").disabled = false; $("#btnRandom").disabled = false; }
}

$("#btnGenerate").onclick = () => runGenerate(false);
$("#btnRandom").onclick = () => runGenerate(true);

// ----- Spotify ---------------------------------------------------------------
let spMode = "existing";

async function loadSpotifyStyles() {
  const styles = await api("GET", "/api/styles");
  const sel = $("#spStyle");
  sel.innerHTML = `<option value="">— choisir un style —</option>` +
    styles.map((s) => `<option value="${s.name}">${s.name} (${s.raw_count} img)</option>`).join("");
  // si aucun style, bascule sur "nouveau"
  if (!styles.length) setSpMode("new");
  // restaure les clés mémorisées
  $("#spId").value = localStorage.getItem("sp_id") || "";
  $("#spSecret").value = localStorage.getItem("sp_secret") || "";
}

function setSpMode(mode) {
  spMode = mode;
  $("#spModeExisting").classList.toggle("active", mode === "existing");
  $("#spModeNew").classList.toggle("active", mode === "new");
  $("#spExistingPane").style.display = mode === "existing" ? "block" : "none";
  $("#spNewPane").style.display = mode === "new" ? "block" : "none";
}
$("#spModeExisting").onclick = () => setSpMode("existing");
$("#spModeNew").onclick = () => setSpMode("new");

$("#spSaveKeys").onclick = () => {
  localStorage.setItem("sp_id", $("#spId").value.trim());
  localStorage.setItem("sp_secret", $("#spSecret").value.trim());
  toast("Clés mémorisées dans ce navigateur.");
};

$("#btnSpotify").onclick = async () => {
  const body = {
    url: $("#spUrl").value.trim(),
    client_id: $("#spId").value.trim(),
    client_secret: $("#spSecret").value.trim(),
    with_artist: $("#spArtist").checked,
    hq: $("#spHq").checked,
  };
  if (spMode === "new") {
    body.new_style = $("#spNewName").value.trim();
    body.new_title_text = $("#spNewTitle").value;
    if (!body.new_style) return toast("Donne un nom au nouveau style.", true);
  } else {
    body.style = $("#spStyle").value;
    if (!body.style) return toast("Choisis un style existant.", true);
  }
  if (!body.url) return toast("Colle un lien de playlist.", true);
  if (!body.client_id || !body.client_secret) return toast("Renseigne tes clés Spotify (à droite).", true);
  // mémorise les clés au passage
  localStorage.setItem("sp_id", body.client_id);
  localStorage.setItem("sp_secret", body.client_secret);
  try {
    const job = await api("POST", "/api/spotify/download", body);
    $("#btnSpotify").disabled = true;
    pollJob(job.id, {
      logEl: $("#spLog"), statusEl: $("#spStatus"),
      onDone: (j) => {
        $("#btnSpotify").disabled = false;
        toast(j.status === "done" ? "Téléchargement terminé ✓" : "Téléchargement : " + j.status, j.status !== "done");
        loadStyles();
        loadSpotifyStyles();
        if (j.status === "done" && j.style) { setSpMode("existing"); $("#spStyle").value = j.style; }
      },
    });
  } catch (e) { toast(e.message, true); $("#btnSpotify").disabled = false; }
};

// ----- Gallery ---------------------------------------------------------------
async function loadGallery() {
  const imgs = await api("GET", "/api/outputs");
  const grid = $("#galleryGrid");
  $("#galleryEmpty").style.display = imgs.length ? "none" : "block";
  grid.innerHTML = imgs.map((i) => `<img src="${i.url}" title="${i.name}" onclick="window.open('${i.url}')" />`).join("");
}

// ----- Init ------------------------------------------------------------------
async function init() {
  await loadStyles();
  // Reprend le suivi si une tâche tourne déjà (rechargement de page)
  try {
    const a = await api("GET", "/api/active");
    if (a.active) setBadge(true, `${a.active.kind} : ${a.active.style || ""}`);
  } catch (e) {}
}
init();
