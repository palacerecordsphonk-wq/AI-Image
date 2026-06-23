# 🎨 Guide : faire apprendre tout l'univers d'un style

Objectif : que le LoRA capte **l'ADN visuel complet** d'un style (couleurs,
atmosphère, grain, composition, typographie) et sache **recréer du neuf**,
inédit mais parfaitement cohérent.

---

## 1. Comment l'IA apprend un style

Le mécanisme central est le **mot-déclencheur** (`trigger` dans `style.yaml`).

Pendant l'entraînement, le LoRA associe ce mot à **tout ce qui est commun** à tes
images. Règle d'or du captioning :

> 🔑 Ce que tu **ne décris pas** est absorbé par le trigger → devient *le style*.
> Ce que tu **décris** devient une *variable modifiable* à la génération.

Donc pour un style, on légende **peu** : juste `trigger + class_word`. Le modèle
n'a alors « rien d'autre à quoi rattacher » les couleurs/ambiance/grain → il les
fond dans le trigger. C'est exactement ce qu'on veut.

À la génération, écrire le trigger en tête du prompt **rappelle tout cet univers**,
et le reste du prompt (titre, ambiance demandée) ne fait que le moduler.

---

## 2. Curation du dataset (le plus important)

La qualité du LoRA = la qualité et la cohérence de tes images. Plus que le nombre.

- **Cohérence avant tout.** Toutes les images d'un dossier doivent partager le
  même ADN (palette, traitement, ambiance). Si deux sous-familles existent
  (ex. covers claires *et* sombres), fais **deux styles séparés**.
- **Quantité** : 15 minimum, **30–150 idéal**. Au-delà de ~200, gains marginaux.
- **Variété dans la cohérence** : varie les sujets/compos *à l'intérieur* du même
  style. Si toutes les images montrent la même chose, le LoRA copie au lieu de
  généraliser (overfitting).
- **Qualité** : images nettes, pas de watermark, pas de doublons. Une mauvaise
  image pollue plus qu'elle n'aide.
- **Texte** : voir section 4.

Le script `prepare_dataset.py` normalise tout (RGB, redimensionnement, renommage).

---

## 3. Deux stratégies de captioning

### Mode SIMPLE (défaut) — recommandé pour démarrer
Légende identique et minimale sur toutes les images :
`"<trigger>, <class_word>"`. Maximise la cohésion : tout l'univers se fond dans
le trigger. Parfait pour un style très homogène.

```bash
python scripts/prepare_dataset.py phonk           # légendes simples auto
```

### Mode AUTO (`--auto`) — pour les styles variés / texte
Un VLM (Qwen2.5-VL via `mlx-vlm`) décrit **chaque** image (couleurs, sujet,
compo) et détecte le titre. La légende garde toujours le trigger en tête mais
ajoute la description → le LoRA distingue mieux les variations internes et tu peux
les **piloter** ensuite au prompt (« neon », « foggy »…).

```bash
pip install mlx-vlm
python scripts/caption.py phonk --auto --overwrite
```

**Quand utiliser quoi ?**
| Situation | Mode |
|---|---|
| Style très homogène, tu veux « le » look | Simple |
| Style avec sous-variations à pouvoir invoquer | Auto |
| Covers avec titres écrits (apprendre la typo + le texte) | Auto |

Tu peux toujours **éditer les `.txt`** dans `dataset/` à la main avant d'entraîner.

---

## 4. Titre & typographie

Réglé par `title_text` dans `style.yaml` :

| Valeur | Le LoRA apprend… | À la génération (défaut) |
|---|---|---|
| `no` | l'absence de texte (look « sans titre ») | n'écrit pas le titre |
| `yes` | la **typographie** du style (police, place, style) | écrit le titre dans ce style |
| `mixed` | les deux (mieux avec `--auto`) | écrit le titre |

- **Style sans titre** (`no`) : le LoRA produit des covers propres sans texte, et
  `generate_cover.py` n'écrit pas le titre (le titre sert juste de thème).
- **Style avec titre** (`yes`) : entraîne avec des covers **qui ont leur texte**.
  Le LoRA apprend *comment* ce style écrit ses titres. En `--auto`, il transcrit
  même le texte réel → meilleur rendu typographique.
- Override ponctuel à la génération : `--title-text` (forcer le texte) ou
  `--no-title-text` (forcer sans).

> 💡 FLUX.1 rend bien le texte court. Pour un rendu typographique fiable, le mode
> `--auto` + `title_text: yes` est nettement supérieur.

---

## 5. Réglages d'entraînement selon la complexité

`train_style.py` calcule les epochs automatiquement (vise `--total-steps`
itérations). Points de départ conseillés :

| Type de style | `--rank` | `--total-steps` | `--max-resolution` |
|---|---|---|---|
| Simple/cohérent (palette + ambiance) | 8–16 | 1000–1200 | 768 (rapide) |
| Moyen | 16 | 1200–1600 | 1024 |
| Complexe (typo + compos variées) | 24–32 | 1600–2400 | 1024 |

- **Trop appris (overfitting)** : les covers ressemblent trop aux images d'entraînement,
  peu de créativité → baisse `--total-steps`, augmente la variété du dataset, ou
  baisse `--lora-scale` à la génération.
- **Pas assez appris** : style peu présent → augmente `--total-steps` ou `--rank`,
  ou monte `--lora-scale` (1.1–1.2).

---

## 6. Vérifier et ajuster

- Pendant l'entraînement, mflux génère des **aperçus** (via `preview.txt`) et une
  courbe de perte dans `styles/<style>/training/`. Regarde si le style apparaît.
- Plusieurs **checkpoints** sont sauvegardés : teste-en plusieurs, le dernier
  n'est pas toujours le meilleur (`generate_cover.py --checkpoint <...>`).
- Joue sur **`--lora-scale`** (0.7 → 1.2) pour doser la force du style sans
  réentraîner.
- Fixe **`--seed`** pour comparer des réglages à composition égale.

---

## 7. Mélanger deux styles

mflux accepte plusieurs LoRA. Pour un crossover (ex. 70 % phonk + 30 % vaporwave),
extrais les deux adaptateurs et appelle `mflux-generate` directement :

```bash
mflux-generate --model dev -q 8 \
  --lora-paths phonk_adapter.safetensors vapor_adapter.safetensors \
  --lora-scales 0.7 0.4 \
  --prompt 'phonk_style, vague_style, album cover, neon night' \
  --steps 25 --guidance 3.5 --width 1024 --height 1024 --output mix.png
```

(les `.safetensors` sont générés par `generate_cover.py` à côté des checkpoints.)

---

## 8. Erreurs fréquentes

- ❌ **Dossier fourre-tout** : mélanger des esthétiques différentes → style flou.
  ✅ Un style = un ADN. Sépare.
- ❌ **Sur-décrire les images** en mode simple → le style ne se fond pas dans le
  trigger. ✅ Garde les légendes minimales (ou passe en `--auto` maîtrisé).
- ❌ **Trop d'images quasi identiques** → copie au lieu de création. ✅ Varie les
  sujets en gardant le traitement.
- ❌ **Trigger trop générique** (un mot anglais courant) → collision avec le savoir
  du modèle. ✅ Un mot inventé (`phonkcover_style`).
