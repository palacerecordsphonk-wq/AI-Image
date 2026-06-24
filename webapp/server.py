#!/usr/bin/env python3
"""Backend de l'interface web locale AI-Image.

Pilote toute la chaîne (création de style, import d'images, préparation +
captioning, entraînement LoRA, génération) en réutilisant les scripts CLI déjà
testés, lancés comme sous-processus avec logs en direct.

Lancer :  python -m webapp.server     (puis ouvrir http://127.0.0.1:8000)
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Rendre les scripts importables (lib.py, etc.)
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import lib  # noqa: E402  (helpers partagés : create_style, import_images, ...)

STATIC = Path(__file__).resolve().parent / "static"
LOGS_DIR = ROOT / ".weblogs"
LOGS_DIR.mkdir(exist_ok=True)

PYTHON = sys.executable

app = FastAPI(title="AI-Image Studio")


# --------------------------------------------------------------------------- #
# Gestion des jobs (sous-processus en arrière-plan avec log fichier)
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    kind: str  # prepare | train | generate
    style: str | None
    cmd: list[str]
    status: str = "running"  # running | done | error | stopped
    returncode: int | None = None
    log_path: Path = field(default=None)  # type: ignore[assignment]
    result: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    ended: float | None = None
    _proc: object = None


class JobManager:
    HEAVY = {"prepare", "train", "generate"}

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def active_heavy(self) -> Job | None:
        for job in self.jobs.values():
            if job.kind in self.HEAVY and job.status == "running":
                return job
        return None

    def start(self, kind: str, cmd: list[str], style: str | None = None,
              on_done=None, env: dict | None = None) -> Job:
        import os
        import subprocess

        # Variables d'env supplémentaires (ex. clé API) — JAMAIS écrites dans le log.
        run_env = {**os.environ, **env} if env else None

        with self._lock:
            # Seules les tâches "lourdes" (GPU) sont mutuellement exclusives.
            # Un téléchargement peut tourner en parallèle.
            if kind in self.HEAVY:
                busy = self.active_heavy()
                if busy is not None:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Une tâche est déjà en cours ({busy.kind}). Attends la fin.",
                    )
            job_id = uuid.uuid4().hex[:12]
            log_path = LOGS_DIR / f"{job_id}.log"
            job = Job(id=job_id, kind=kind, style=style, cmd=cmd, log_path=log_path)
            self.jobs[job_id] = job

        def _run() -> None:
            with open(log_path, "w", encoding="utf-8") as logf:
                logf.write(f"$ {' '.join(cmd)}\n\n")
                logf.flush()
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                        text=True, env=run_env,
                    )
                    job._proc = proc
                    rc = proc.wait()
                except Exception as exc:  # binaire introuvable, etc.
                    logf.write(f"\n[ERREUR] {exc}\n")
                    job.status = "error"
                    job.ended = time.time()
                    return
            job.returncode = rc
            if job.status == "stopped":
                pass
            elif rc == 0:
                job.status = "done"
            else:
                job.status = "error"
            job.ended = time.time()
            if on_done and job.status == "done":
                try:
                    on_done(job)
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()
        return job

    def stop(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job or job.status != "running":
            return
        job.status = "stopped"
        proc = job._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass


jobs = JobManager()


def job_dict(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "style": job.style,
        "status": job.status,
        "returncode": job.returncode,
        "result": job.result,
        "started": job.started,
        "ended": job.ended,
    }


def read_log(job: Job, tail: int = 20000) -> str:
    if job.log_path and job.log_path.exists():
        text = job.log_path.read_text(encoding="utf-8", errors="replace")
        return text[-tail:]
    return ""


# --------------------------------------------------------------------------- #
# Modèles de requête
# --------------------------------------------------------------------------- #
class CreateStyle(BaseModel):
    name: str
    class_word: str | None = None
    title_text: str = "mixed"
    base_model: str = "dev"
    description: str = ""


class ImportPath(BaseModel):
    path: str


class PrepareReq(BaseModel):
    size: int = 1024
    # Backend de captioning : "simple" (sans IA), "local" (mlx-vlm) ou "gemini" (API).
    caption_backend: str = "local"
    gemini_api_key: str | None = None
    # Rétro-compat : ancien booléen (true -> local). Ignoré si caption_backend fourni.
    auto_caption: bool | None = None


class TrainReq(BaseModel):
    max_resolution: int = 1024
    total_steps: int = 1200
    rank: int = 16
    quantize: int | None = 8
    model: str | None = None  # clé du registre (flux1-dev, flux2-klein-9b, …)


class SpotifyReq(BaseModel):
    url: str
    client_id: str
    client_secret: str
    # Destination = le dossier raw/ d'un style. Soit un style existant, soit un
    # nouveau à créer.
    style: str | None = None          # style existant
    new_style: str | None = None      # nom d'un nouveau style à créer
    new_title_text: str = "mixed"     # convention de titre du nouveau style
    with_artist: bool = True          # nom de fichier "Artiste - Titre" (sinon "Titre")
    hq: bool = True                   # haute résolution via Apple Music (~3000px)


class GenerateReq(BaseModel):
    style: str
    title: str = ""
    extra: str = ""
    title_mode: str = "auto"  # auto | text | notext
    grain: str = "auto"       # auto | none | light | medium | heavy
    random: bool = False      # composer un prompt aléatoire depuis le style appris
    model: str | None = None  # clé du registre (flux1-dev, flux2-klein-9b, …)
    steps: int = 25
    guidance: float = 3.5
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    quantize: int | None = 8
    lora_scale: float = 1.0


# --------------------------------------------------------------------------- #
# Endpoints : styles
# --------------------------------------------------------------------------- #
@app.get("/api/models")
def api_models() -> dict:
    """Liste les modèles de base disponibles (FLUX.1 / FLUX.2) pour l'UI."""
    return {
        "default": lib.DEFAULT_MODEL,
        "models": [
            {"key": k, "label": v["label"], "family": v["family"]}
            for k, v in lib.MODELS.items()
        ],
    }


@app.get("/api/styles")
def api_styles() -> list[dict]:
    out = []
    for name in lib.list_styles():
        cfg = lib.load_style(name)
        sdir = lib.style_dir(name)
        raw = lib.find_images(sdir / "raw")
        dataset = lib.find_images(sdir / "dataset")
        trained = lib.trained_models(name)
        out.append({
            "name": name,
            "trigger": cfg["trigger"],
            "class_word": cfg["class_word"],
            "title_text": cfg["title_text"],
            "base_model": cfg["base_model"],
            "raw_count": len(raw),
            "dataset_count": len(dataset),
            "trained": bool(trained),
            "trained_models": trained,  # clés des modèles ayant un checkpoint
            "raw_path": str((sdir / "raw").resolve()),
        })
    return out


@app.post("/api/styles")
def api_create_style(req: CreateStyle) -> dict:
    name = lib.slugify(req.name)
    if not name:
        raise HTTPException(400, "Nom de style invalide.")
    if name in lib.list_styles():
        raise HTTPException(409, f"Le style '{name}' existe déjà.")
    title_text = req.title_text if req.title_text in {"yes", "no", "mixed"} else "mixed"
    lib.create_style(
        name,
        description=req.description,
        class_word=req.class_word or f"{name} album cover",
        title_text=title_text,
        base_model=req.base_model,
    )
    return {"ok": True, "name": name}


@app.post("/api/styles/{name}/import-path")
def api_import_path(name: str, req: ImportPath) -> dict:
    _require_style(name)
    src = Path(req.path.strip().strip('"').strip("'").replace("\\ ", " ")).expanduser()
    if not src.exists() or not src.is_dir():
        raise HTTPException(400, f"Dossier introuvable : {src}")
    n = lib.import_images(src, lib.style_dir(name) / "raw")
    return {"ok": True, "imported": n}


@app.post("/api/styles/{name}/upload")
async def api_upload(name: str, files: list[UploadFile]) -> dict:
    _require_style(name)
    raw_dir = lib.style_dir(name) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in lib.IMAGE_EXTS:
            continue
        target = raw_dir / Path(f.filename).name
        i = 1
        while target.exists():
            target = raw_dir / f"{Path(f.filename).stem}_{i}{ext}"
            i += 1
        with open(target, "wb") as out:
            out.write(await f.read())
        saved += 1
    return {"ok": True, "imported": saved}


@app.get("/api/styles/{name}/images")
def api_style_images(name: str) -> dict:
    _require_style(name)
    sdir = lib.style_dir(name)
    dataset = lib.find_images(sdir / "dataset")
    raw = lib.find_images(sdir / "raw")
    imgs = dataset if dataset else raw
    folder = "dataset" if dataset else "raw"
    items = []
    for p in imgs[:60]:
        caption = ""
        if folder == "dataset":
            # La légende d'entraînement est le .txt à côté de l'image préparée.
            txt = p.with_suffix(".txt")
            if txt.exists():
                caption = txt.read_text(encoding="utf-8", errors="replace").strip()
        items.append({"url": f"/media/{name}/{folder}/{p.name}", "caption": caption})
    return {
        "folder": folder,
        "items": items,
        # Rétro-compat : ancienne clé "images" (URLs seules).
        "images": [it["url"] for it in items],
        "total": len(imgs),
    }


@app.get("/media/{name}/{folder}/{filename}")
def api_media(name: str, folder: str, filename: str):
    if folder not in {"raw", "dataset"}:
        raise HTTPException(404, "not found")
    path = (lib.style_dir(name) / folder / filename).resolve()
    base = lib.style_dir(name).resolve()
    if base not in path.parents or not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# Endpoints : jobs (prepare / train / generate)
# --------------------------------------------------------------------------- #
@app.post("/api/styles/{name}/prepare")
def api_prepare(name: str, req: PrepareReq) -> dict:
    _require_style(name)
    # Détermine le backend (caption_backend prioritaire, sinon l'ancien booléen).
    backend = (req.caption_backend or "").strip().lower()
    if backend not in {"simple", "local", "gemini"}:
        backend = "local" if req.auto_caption else "simple"

    cmd = [PYTHON, str(SCRIPTS / "prepare_dataset.py"), name, "--size", str(req.size)]
    env = None
    if backend == "local":
        cmd.append("--auto-caption")
    elif backend == "gemini":
        key = (req.gemini_api_key or "").strip()
        if not key:
            raise HTTPException(400, "Clé API Gemini requise pour le captioning Gemini.")
        cmd.append("--gemini")
        # Clé passée par l'environnement -> jamais visible dans le log de commande.
        env = {"GEMINI_API_KEY": key}
    # backend == "simple" : aucune option (légendes minimales).

    job = jobs.start("prepare", cmd, style=name, env=env)
    return job_dict(job)


@app.post("/api/styles/{name}/train")
def api_train(name: str, req: TrainReq) -> dict:
    _require_style(name)
    cmd = [
        PYTHON, str(SCRIPTS / "train_style.py"), name,
        "--max-resolution", str(req.max_resolution),
        "--total-steps", str(req.total_steps),
        "--rank", str(req.rank),
    ]
    if req.model:
        cmd += ["--model", lib.resolve_model_key(req.model)]
    if req.quantize is None:
        cmd.append("--no-quantize")
    else:
        cmd += ["--quantize", str(req.quantize)]
    job = jobs.start("train", cmd, style=name)
    return job_dict(job)


@app.post("/api/generate")
def api_generate(req: GenerateReq) -> dict:
    _require_style(req.style)
    out_name = f"{req.style}_{int(time.time())}.png"
    out_path = lib.OUTPUT_DIR / out_name
    cmd = [
        PYTHON, str(SCRIPTS / "generate_cover.py"), req.style,
        "--title", req.title,
        "--extra", req.extra,
        "--steps", str(req.steps),
        "--guidance", str(req.guidance),
        "--width", str(req.width),
        "--height", str(req.height),
        "--lora-scale", str(req.lora_scale),
        "--output", str(out_path),
    ]
    if req.quantize is None:
        cmd.append("--no-quantize")
    else:
        cmd += ["--quantize", str(req.quantize)]
    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]
    if req.title_mode == "text":
        cmd.append("--title-text")
    elif req.title_mode == "notext":
        cmd.append("--no-title-text")
    grain = req.grain if req.grain in {"auto", "none", "light", "medium", "heavy"} else "auto"
    cmd += ["--grain", grain]
    if req.random:
        cmd.append("--random")
    if req.model:
        cmd += ["--model", lib.resolve_model_key(req.model)]

    def _done(job: Job) -> None:
        if out_path.exists():
            job.result = {"image": f"/outputs/{out_name}"}

    job = jobs.start("generate", cmd, style=req.style, on_done=_done)
    return job_dict(job)


@app.post("/api/spotify/download")
def api_spotify(req: SpotifyReq) -> dict:
    if not req.url.strip():
        raise HTTPException(400, "Lien de playlist requis.")
    if not req.client_id.strip() or not req.client_secret.strip():
        raise HTTPException(400, "Client ID et Client Secret Spotify requis.")

    # Déterminer le style cible (existant ou nouveau) -> destination = son raw/.
    if req.new_style and req.new_style.strip():
        name = lib.slugify(req.new_style)
        if not name:
            raise HTTPException(400, "Nom de nouveau style invalide.")
        if name in lib.list_styles():
            raise HTTPException(409, f"Le style '{name}' existe déjà.")
        title_text = req.new_title_text if req.new_title_text in {"yes", "no", "mixed"} else "mixed"
        lib.create_style(name, class_word=f"{name} album cover", title_text=title_text)
        target = name
    elif req.style and req.style.strip():
        target = req.style.strip()
        _require_style(target)
    else:
        raise HTTPException(400, "Choisis un style existant ou crée un nouveau style.")

    dest = lib.style_dir(target) / "raw"
    cmd = [
        PYTHON, str(SCRIPTS / "spotify_covers.py"), req.url.strip(),
        "--dest", str(dest),
        "--client-id", req.client_id.strip(),
        "--client-secret", req.client_secret.strip(),
    ]
    if not req.with_artist:
        cmd.append("--no-artist")
    if not req.hq:
        cmd.append("--no-hq")
    job = jobs.start("download", cmd, style=target)
    return job_dict(job)


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    job = jobs.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job introuvable")
    d = job_dict(job)
    d["log"] = read_log(job)
    return d


@app.post("/api/jobs/{job_id}/stop")
def api_job_stop(job_id: str) -> dict:
    jobs.stop(job_id)
    return {"ok": True}


@app.get("/api/active")
def api_active() -> dict:
    job = jobs.active_heavy()
    return {"active": job_dict(job) if job else None}


# --------------------------------------------------------------------------- #
# Endpoints : galerie de sorties
# --------------------------------------------------------------------------- #
@app.get("/api/outputs")
def api_outputs() -> list[dict]:
    lib.OUTPUT_DIR.mkdir(exist_ok=True)
    imgs = sorted(lib.OUTPUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"url": f"/outputs/{p.name}", "name": p.name, "mtime": p.stat().st_mtime} for p in imgs]


@app.get("/outputs/{filename}")
def api_output_file(filename: str):
    path = (lib.OUTPUT_DIR / filename).resolve()
    if lib.OUTPUT_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _require_style(name: str) -> None:
    if name not in lib.list_styles():
        raise HTTPException(404, f"Style '{name}' introuvable.")


def main() -> None:
    import uvicorn

    print("\n🎨  AI-Image Studio")
    print("    Ouvre ton navigateur sur :  http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
