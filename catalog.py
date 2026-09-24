"""The object catalog: what the app knows how to look for, and name search.

Two kinds of objects exist:
  * tag     - carries a printed ArUco marker (DICT_4X4_50, IDs 0..49).
              Built-in tags come from objects.json; more can be registered
              from the browser by showing an unused tag to the camera.
  * visual  - enrolled by drawing a box around it (markerless matching).
              Stored under data/templates/, with one or more views.

The catalog file (data/catalog.json) is written atomically. Entries whose
template files are missing are kept and reported as a problem instead of
being silently dropped (the original version silently dropped them and then
reused their ID, so a new object could inherit an old object's history).
"""
from __future__ import annotations

import difflib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TAG_IDS = range(0, 50)           # ArUco DICT_4X4_50
FIRST_VISUAL_ID = 50
MAX_ALIASES = 8
NAME_MAX = 45

# Words that carry no identity in a search ("where are my keys?").
STOPWORDS = {
    'a', 'an', 'the', 'my', 'mine', 'our', 'your', 'his', 'her', 'their', 'is', 'are', 'was', 'were',
    'where', 'wheres', "where's", 'what', 'whats', 'find', 'locate', 'search', 'show', 'me', 'for',
    'did', 'do', 'i', 'we', 'put', 'leave', 'left', 'last', 'seen', 'see', 'can', 'you', 'please',
    'of', 'it', 'at', 'to', 'in', 'on', 'look', 'looking', 'point', 'go', 'get', 'hey', 'ok',
    'okay', 'thing', 'item', 'object', 'again', 'now', 'just', 'go', 'lost', 'missing', 'help',
}


def tokens(text: str) -> list[str]:
    words = re.findall(r"[\w']+", text.casefold())
    out = []
    for word in words:
        word = word.strip("'")
        if word.endswith("'s"):
            word = word[:-2]
        if not word:
            continue
        out.append(_singular(word))
    return out


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith('ies'):
        return word[:-3] + 'y'
    if len(word) > 4 and word.endswith(('ches', 'shes', 'sses', 'xes')):
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith(('ss', 'us', 'is')):
        return word[:-1]
    return word


def key_tokens(text: str) -> list[str]:
    """Tokens that identify an object; falls back to all tokens for names
    made only of stop-words."""
    all_tokens = tokens(text)
    meaningful = [t for t in all_tokens if t not in STOPWORDS]
    return meaningful or all_tokens


@dataclass
class CatalogObject:
    id: int
    name: str
    aliases: list[str] = field(default_factory=list)
    kind: str = 'visual'            # 'tag' or 'visual'
    builtin: bool = False
    views: list[str] = field(default_factory=list)
    created: float | None = None
    problem: str | None = None      # set at runtime when unusable

    @property
    def phrases(self) -> list[str]:
        return [self.name, *self.aliases]

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {'name': self.name, 'aliases': self.aliases, 'kind': self.kind}
        if self.views:
            data['views'] = self.views
        if self.created:
            data['created'] = round(self.created, 3)
        return data


def clean_name(name: str) -> str:
    name = ' '.join(str(name).split())
    if not 2 <= len(name) <= NAME_MAX:
        raise ValueError(f'Names must be 2-{NAME_MAX} characters long.')
    if not key_tokens(name):
        raise ValueError('Use letters or numbers in the name.')
    return name


def clean_aliases(aliases) -> list[str]:
    cleaned, seen = [], set()
    for alias in aliases or []:
        alias = ' '.join(str(alias).split())[:NAME_MAX]
        if len(alias) < 2 or not key_tokens(alias) or alias.casefold() in seen:
            continue
        seen.add(alias.casefold())
        cleaned.append(alias)
    return cleaned[:MAX_ALIASES]


def load_builtin(path: Path) -> dict[int, CatalogObject]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    objects: dict[int, CatalogObject] = {}
    for key, item in raw.items():
        marker = int(key)
        if marker not in TAG_IDS:
            raise ValueError('Marker IDs must be between 0 and 49 for DICT_4X4_50')
        if not isinstance(item.get('name'), str) or not item['name'].strip():
            raise ValueError(f'Object {marker} must have a name')
        objects[marker] = CatalogObject(marker, item['name'].strip(), clean_aliases(item.get('aliases', [])),
                                        'tag', builtin=True)
    return objects


class Catalog:
    def __init__(self, builtin_path: Path, catalog_path: Path):
        self.builtin_path = builtin_path
        self.path = catalog_path
        self.objects: dict[int, CatalogObject] = {}
        self.next_id = FIRST_VISUAL_ID
        self.load()

    # ---- persistence -------------------------------------------------------
    def load(self) -> None:
        objects = load_builtin(self.builtin_path)
        next_id = FIRST_VISUAL_ID
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding='utf-8'))
            except (ValueError, OSError) as exc:
                raise RuntimeError(f'Cannot read {self.path}: {exc}. Fix or move the file and restart.') from exc
            if isinstance(raw, dict) and raw.get('version') == 2:
                entries = raw.get('objects', {})
                next_id = int(raw.get('next_id', FIRST_VISUAL_ID))
            elif isinstance(raw, dict):
                # Version-1 format: {"50": {"name": ..., "aliases": [...]}}
                entries = {k: {**v, 'kind': 'visual', 'views': [f'{int(k)}.png']} for k, v in raw.items()}
            else:
                raise RuntimeError(f'Unexpected content in {self.path}.')
            for key, item in entries.items():
                try:
                    obj_id = int(key)
                    kind = item.get('kind', 'visual')
                    if kind == 'tag' and obj_id not in TAG_IDS:
                        raise ValueError('bad tag id')
                    if kind == 'visual' and obj_id < FIRST_VISUAL_ID:
                        raise ValueError('bad visual id')
                    if obj_id in objects:
                        continue  # never shadow a built-in tag
                    objects[obj_id] = CatalogObject(
                        obj_id, str(item['name']), clean_aliases(item.get('aliases', [])), kind,
                        views=[str(v) for v in item.get('views', [])] if kind == 'visual' else [],
                        created=item.get('created'))
                except (ValueError, TypeError, KeyError, AttributeError):
                    continue
        visual_ids = [i for i, o in objects.items() if o.kind == 'visual']
        self.next_id = max([next_id, *(i + 1 for i in visual_ids)])
        self.objects = objects

    def save(self) -> None:
        data = {'version': 2, 'next_id': self.next_id,
                'objects': {str(i): o.to_json() for i, o in sorted(self.objects.items()) if not o.builtin}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    # ---- queries -----------------------------------------------------------
    def get(self, obj_id: int) -> CatalogObject | None:
        return self.objects.get(obj_id)

    def tag_ids(self) -> set[int]:
        return {i for i, o in self.objects.items() if o.kind == 'tag'}

    def visual(self) -> dict[int, CatalogObject]:
        return {i: o for i, o in self.objects.items() if o.kind == 'visual'}

    def check_names(self, name: str, aliases: list[str], exclude: int | None = None) -> None:
        wanted = {' '.join(key_tokens(p)) for p in [name, *aliases]}
        for obj_id, obj in self.objects.items():
            if obj_id == exclude:
                continue
            for phrase in obj.phrases:
                if ' '.join(key_tokens(phrase)) in wanted:
                    raise ValueError(f'"{phrase}" is already used by {obj.name}. Choose a different name or alias.')

    # ---- mutations (caller persists with save()) ----------------------------
    def add_tag(self, marker_id: int, name: str, aliases=()) -> CatalogObject:
        if marker_id not in TAG_IDS:
            raise ValueError('Tag numbers run from 0 to 49.')
        if marker_id in self.objects:
            raise ValueError(f'Tag #{marker_id} is already registered as {self.objects[marker_id].name}.')
        name, aliases = clean_name(name), clean_aliases(aliases)
        self.check_names(name, aliases)
        obj = CatalogObject(marker_id, name, aliases, 'tag', created=time.time())
        self.objects[marker_id] = obj
        return obj

    def add_visual(self, name: str, aliases=()) -> CatalogObject:
        name, aliases = clean_name(name), clean_aliases(aliases)
        self.check_names(name, aliases)
        obj_id = self.next_id
        self.next_id += 1              # IDs are never reused
        obj = CatalogObject(obj_id, name, aliases, 'visual', created=time.time())
        self.objects[obj_id] = obj
        return obj

    def remove(self, obj_id: int) -> CatalogObject:
        obj = self.objects.get(obj_id)
        if obj is None:
            raise KeyError(obj_id)
        if obj.builtin:
            raise ValueError('Built-in tags stay in the library; only their history can be cleared.')
        return self.objects.pop(obj_id)

    # ---- search ------------------------------------------------------------
    def search(self, query: str, usable_only: bool = False) -> list[tuple[float, int]]:
        return search(query, {i: o for i, o in self.objects.items() if not (usable_only and o.problem)})


def _score(query_tokens: list[str], phrase: str) -> float:
    phrase_tokens = key_tokens(phrase)
    if not phrase_tokens:
        return 0.0
    qset = set(query_tokens)
    present = [t for t in phrase_tokens if t in qset]
    if len(present) == len(phrase_tokens):
        # Whole phrase present as words. Longer phrases are more specific.
        return 100 + 10 * len(phrase_tokens) + len(''.join(phrase_tokens)) * .1
    fuzzy = 0.0
    for token in phrase_tokens:
        if token in qset:
            fuzzy += 1
            continue
        best = max((difflib.SequenceMatcher(None, token, q).ratio() for q in query_tokens), default=0)
        if best >= .8 and len(token) >= 4:
            fuzzy += best          # tolerate typos such as "earphnes"
    fraction = fuzzy / len(phrase_tokens)
    if len(phrase_tokens) == 1:
        return 60 * fraction if fraction >= .8 else 0.0
    return 50 * fraction if fraction >= .5 else 0.0


def search(query: str, objects: dict[int, CatalogObject]) -> list[tuple[float, int]]:
    """Rank objects for a free-text query. Local and auditable: word matching
    with plural folding and small typo tolerance. No network, no AI model."""
    query_tokens = key_tokens(query)
    if not query_tokens:
        return []
    ranked = []
    for obj_id, obj in objects.items():
        score = max((_score(query_tokens, phrase) for phrase in obj.phrases), default=0.0)
        if score > 0:
            ranked.append((round(score, 3), obj_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked
