#!/usr/bin/env python3
"""Télécharge en masse les covers (pochettes) d'une playlist Spotify.

Donne un lien de playlist + un dossier de destination : récupère TOUS les titres
de la playlist et télécharge la cover de chacun, nommée d'après le titre du
morceau.

Authentification : nécessite des identifiants développeur Spotify (gratuits) via
le flux "Client Credentials" — aucune connexion à ton compte requise.
  1. Va sur https://developer.spotify.com/dashboard
  2. "Create app" -> récupère le Client ID et le Client Secret
  3. Fournis-les via --client-id / --client-secret ou les variables
     d'environnement SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET.

Utilisation :
    python scripts/spotify_covers.py "https://open.spotify.com/playlist/XXXX" \
        --dest "/Users/moi/Desktop/covers_funk" \
        --client-id ID --client-secret SECRET

    python scripts/spotify_covers.py --self-test   # vérifie les fonctions internes
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"
ITUNES_SEARCH = "https://itunes.apple.com/search"
IMAGE_EXTS_OK = {".jpg", ".jpeg", ".png", ".webp"}


# --------------------------------------------------------------------------- #
# Fonctions pures (testables hors-ligne)
# --------------------------------------------------------------------------- #
def parse_playlist_id(url_or_id: str) -> str:
    """Extrait l'ID de playlist depuis une URL/URI/ID Spotify."""
    s = url_or_id.strip()
    # spotify:playlist:ID
    m = re.search(r"playlist[:/]([A-Za-z0-9]+)", s)
    if m:
        return m.group(1)
    # déjà un ID nu ?
    if re.fullmatch(r"[A-Za-z0-9]{16,}", s):
        return s
    raise ValueError(f"Lien de playlist Spotify invalide : {url_or_id!r}")


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Nettoie un titre pour en faire un nom de fichier sûr."""
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "", name)   # caractères interdits
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .")
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "cover"


def pick_largest_image(images: list[dict]) -> str | None:
    """Retourne l'URL de la plus grande image d'une liste Spotify."""
    if not images:
        return None
    best = max(images, key=lambda im: (im.get("width") or 0) * (im.get("height") or 0))
    return best.get("url")


def itunes_upscale_url(url: str, res: int) -> str:
    """Transforme une URL d'artwork iTunes (…/100x100bb.jpg) en haute résolution.

    Apple sert l'image à la taille demandée (jusqu'à la résolution source, souvent
    jusqu'à ~3000px). Ex : …/100x100bb.jpg -> …/3000x3000bb.jpg
    """
    return re.sub(r"/\d+x\d+bb\.(jpg|jpeg|png)", f"/{res}x{res}bb.\\1", url)


def existing_with_stem(dest: Path, stem: str) -> bool:
    """Vrai si une image de même nom (toute extension) existe déjà -> doublon."""
    return any((dest / f"{stem}{ext}").exists() for ext in IMAGE_EXTS_OK)


# --------------------------------------------------------------------------- #
# Appels réseau Spotify
# --------------------------------------------------------------------------- #
def get_token(client_id: str, client_secret: str) -> str:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=data,
        headers={"Authorization": f"Basic {auth}",
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["access_token"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Échec d'authentification Spotify ({e.code}). "
            f"Vérifie ton Client ID / Client Secret."
        ) from e


def _api_get(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def iter_playlist_tracks(playlist_id: str, token: str):
    """Génère (titre, artistes, album, url_cover_spotify) pour chaque morceau."""
    fields = "next,total,items(track(name,artists(name),album(name,images)))"
    offset = 0
    limit = 100
    while True:
        url = (f"{API}/playlists/{playlist_id}/tracks?"
               + urllib.parse.urlencode({"limit": limit, "offset": offset,
                                         "fields": fields, "additional_types": "track"}))
        data = _api_get(url, token)
        items = data.get("items", [])
        if not items:
            break
        for it in items:
            track = it.get("track") or {}
            name = track.get("name")
            if not name:
                continue
            artists = ", ".join(a.get("name", "") for a in track.get("artists", []) if a.get("name"))
            album = track.get("album") or {}
            cover = pick_largest_image(album.get("images", []))
            yield name, artists, album.get("name", ""), cover
        offset += limit
        if not data.get("next"):
            break


def itunes_hires_url(artist: str, album: str, res: int) -> str | None:
    """Cherche la cover de l'album sur iTunes et renvoie une URL haute résolution.

    Gratuit, sans authentification. Renvoie None si rien trouvé.
    """
    term = " ".join(p for p in [artist.split(",")[0].strip(), album] if p).strip()
    if not term:
        return None
    url = ITUNES_SEARCH + "?" + urllib.parse.urlencode(
        {"term": term, "entity": "album", "limit": 5}
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AI-Image/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            results = json.loads(r.read()).get("results", [])
    except Exception:
        return None
    if not results:
        return None

    # Choisit le meilleur résultat : album dont le nom correspond le mieux.
    alb_low = album.strip().lower()
    best = None
    for res_item in results:
        coll = (res_item.get("collectionName") or "").lower()
        if alb_low and (alb_low in coll or coll in alb_low):
            best = res_item
            break
    best = best or results[0]
    art = best.get("artworkUrl100") or best.get("artworkUrl60")
    return itunes_upscale_url(art, res) if art else None


def download_image(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Image/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(target, "wb") as f:
        f.write(r.read())


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(url: str, dest: Path, client_id: str, client_secret: str, *,
        with_artist: bool = True, hq: bool = True, res: int = 3000,
        itunes_delay: float = 0.3) -> int:
    playlist_id = parse_playlist_id(url)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"🎧 Playlist : {playlist_id}")
    print(f"📁 Destination : {dest}")
    print(f"🖼️  Qualité : {'Apple Music HD (~' + str(res) + 'px)' if hq else 'Spotify (640px)'}")

    token = get_token(client_id, client_secret)
    print("✅ Authentifié auprès de Spotify.\n")

    downloaded = 0
    skipped = 0
    duplicates = 0
    hires_hits = 0
    seen_stems: set[str] = set()  # doublons à l'intérieur du même téléchargement
    itunes_cache: dict[tuple, str | None] = {}  # (artist, album) -> url HD (ou None)
    tracks = list(iter_playlist_tracks(playlist_id, token))
    total = len(tracks)
    print(f"🔎 {total} titre(s) trouvé(s).\n")

    for i, (name, artists, album, spotify_cover) in enumerate(tracks, 1):
        label = f"{artists} - {name}" if (with_artist and artists) else name
        stem = sanitize_filename(label)

        # Dédoublonnage : même morceau déjà présent (téléchargement précédent OU
        # ce téléchargement) -> on saute, pas de re-téléchargement.
        if stem in seen_stems or existing_with_stem(dest, stem):
            print(f"[{i}/{total}] ♻️  doublon ignoré : {stem}")
            duplicates += 1
            seen_stems.add(stem)
            continue

        # 1) Tente la haute résolution via iTunes (avec cache par album).
        cover = None
        is_hd = False
        if hq:
            key = (artists.split(",")[0].strip().lower(), album.strip().lower())
            if key not in itunes_cache:
                itunes_cache[key] = itunes_hires_url(artists, album, res)
                time.sleep(itunes_delay)  # respecte la limite de l'API iTunes
            cover = itunes_cache[key]
            is_hd = cover is not None

        # 2) Repli sur la cover Spotify (640px) si pas de HD.
        if not cover:
            cover = spotify_cover

        if not cover:
            print(f"[{i}/{total}] ⏭️  {label} (pas de cover)")
            skipped += 1
            continue

        ext = os.path.splitext(urllib.parse.urlparse(cover).path)[1].lower()
        if ext not in IMAGE_EXTS_OK:
            ext = ".jpg"
        target = dest / f"{stem}{ext}"
        try:
            download_image(cover, target)
            downloaded += 1
            seen_stems.add(stem)
            if is_hd:
                hires_hits += 1
            tag = "🟢 HD" if is_hd else "⚪ 640"
            print(f"[{i}/{total}] ⬇️  {tag}  {target.name}")
        except Exception as exc:
            skipped += 1
            print(f"[{i}/{total}] ⚠️  Échec {label} : {exc}")

    print(f"\n🎉 Terminé : {downloaded} cover(s) téléchargée(s) "
          f"({hires_hits} en HD, {downloaded - hires_hits} en 640px), "
          f"{duplicates} doublon(s) ignoré(s), {skipped} sans cover.")
    print(f"   Dans : {dest}")
    return 0


def _self_test() -> None:
    assert parse_playlist_id("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M") == "37i9dQZF1DXcBWIGoYBM5M"
    assert parse_playlist_id("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M") == "37i9dQZF1DXcBWIGoYBM5M"
    assert parse_playlist_id("https://open.spotify.com/playlist/ABC123?si=xyz") == "ABC123"
    assert sanitize_filename('AC/DC: Thunder*?') == "ACDC Thunder"
    assert sanitize_filename("   ") == "cover"
    assert pick_largest_image([{"url": "a", "width": 64, "height": 64},
                               {"url": "b", "width": 640, "height": 640}]) == "b"
    assert pick_largest_image([]) is None
    u = "https://is1-ssl.mzstatic.com/image/thumb/abc/source/100x100bb.jpg"
    assert itunes_upscale_url(u, 3000).endswith("/3000x3000bb.jpg")
    assert itunes_upscale_url(u, 1600).endswith("/1600x1600bb.jpg")
    print("✅ self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharger les covers d'une playlist Spotify.")
    parser.add_argument("url", nargs="?", help="Lien/URI/ID de la playlist Spotify.")
    parser.add_argument("--dest", required=False, help="Dossier de destination.")
    parser.add_argument("--client-id", default=os.environ.get("SPOTIFY_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.environ.get("SPOTIFY_CLIENT_SECRET"))
    parser.add_argument("--no-artist", action="store_true",
                        help="Nommer 'Titre' seul (défaut : 'Artiste - Titre').")
    parser.add_argument("--no-hq", action="store_true",
                        help="Désactiver la haute résolution (garder le 640px de Spotify).")
    parser.add_argument("--res", type=int, default=3000,
                        help="Résolution cible en HD via Apple Music (défaut 3000).")
    parser.add_argument("--itunes-delay", type=float, default=0.3,
                        help="Pause entre recherches iTunes, en s (défaut 0.3).")
    parser.add_argument("--self-test", action="store_true", help="Tester les fonctions internes.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.url or not args.dest:
        parser.error("url et --dest sont requis.")
    if not args.client_id or not args.client_secret:
        parser.error(
            "Identifiants Spotify manquants. Fournis --client-id/--client-secret "
            "ou SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET."
        )

    try:
        rc = run(args.url, Path(args.dest).expanduser(), args.client_id, args.client_secret,
                 with_artist=not args.no_artist, hq=not args.no_hq, res=args.res,
                 itunes_delay=args.itunes_delay)
    except Exception as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
