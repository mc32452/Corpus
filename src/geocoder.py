"""Offline geocoder based on GeoNames cities500.txt.

Attribution: GeoNames (https://www.geonames.org)
License: Creative Commons Attribution 4.0 International
See: https://download.geonames.org/export/dump/readme.txt

Architecture:
- places_by_id: dict[int, GeoPlace]            one record per geonameid
- alias_to_ids: dict[str, list[int]]           normalized alias -> [geonameid]
- _id_to_aliases: dict[int, list[str]]         geonameid -> aliases
- _ngram_to_ids: dict[str, set[int]]           trigram -> geonameid set
- kdtree: cKDTree on 3D unit sphere vectors
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.spatial import cKDTree

from .geo_types import GeoMethod, GeocoderState

log = logging.getLogger(__name__)

# Tunables
_GEONAMES_PATH = "data/cities500.txt"
_VERSION_PATH = "data/geo_version.json"
_NGRAM_SIZE = 3
_NGRAM_CANDIDATES = 1_500
_DEFAULT_THRESHOLD = 72
_AMBIGUITY_WINDOW = 5.0
_STALE_DAYS = 180
_FORWARD_CACHE_SIZE = 1_024
_EARTH_RADIUS_KM = 6_371.0
_GEO_BOOST_MAX = 0.18
_GEO_DECAY_HALF_KM = 25.0
_MAX_EXPANSION_TERMS = 6

_COLS = [
    "geonameid", "name", "asciiname", "altnames", "lat", "lon",
    "feat_class", "feat_code", "country", "cc2",
    "admin1", "admin2", "admin3", "admin4",
    "population", "elevation", "dem", "timezone", "modified",
]

# Named historical regions: (lat, lon, radius_km)
NAMED_REGIONS: dict[str, tuple[float, float, float]] = {
    "mesopotamia": (33.5, 44.4, 350),
    "levant": (33.0, 36.0, 300),
    "rhineland": (50.9, 7.0, 150),
    "iberia": (40.0, -4.0, 700),
    "anatolia": (39.0, 35.0, 600),
    "gaul": (46.5, 2.5, 600),
    "magna graecia": (39.0, 16.5, 350),
    "holy land": (31.7, 35.2, 150),
    "bohemia": (49.8, 15.5, 200),
    "transylvania": (46.5, 24.5, 200),
    "pannonia": (47.0, 18.0, 350),
    "dacia": (45.5, 25.0, 300),
    "numidia": (36.0, 6.0, 400),
    "thrace": (41.5, 26.0, 250),
    "bithynia": (40.5, 30.0, 200),
}

# Query extraction patterns
_SPATIAL_CUES = re.compile(
    r"\b(?:near|around|from|at|within|close to|proximate to|"
    r"surrounding|adjacent to|vicinity of|region of|area of|"
    r"province of|kingdom of)\b\s+",
    re.IGNORECASE,
)
_QUOTED_PLACE = re.compile(r'"([^"]{2,60})"')
_CAP_NP_STRICT = re.compile(r"^([A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2})")
_CAP_NP_CHAIN = re.compile(
    r"^([A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2}(?:\s+and\s+[A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2})*)"
)
_BETWEEN_CHAIN = re.compile(
    r"\bbetween\s+([A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2})\s+and\s+([A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2})",
    re.IGNORECASE,
)

_COUNTRY_HINT_TO_CODE = {
    "france": "FR",
    "french": "FR",
    "united states": "US",
    "usa": "US",
    "us": "US",
    "egypt": "EG",
    "egyptian": "EG",
    "syria": "SY",
    "syrian": "SY",
    "turkey": "TR",
    "turkish": "TR",
}


@dataclass(frozen=True)
class GeoPlace:
    geonameid: int
    name: str
    asciiname: str
    lat: float
    lon: float
    country: str
    admin1: str
    population: int
    top_aliases: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "geonameid": self.geonameid,
            "display": self.name,
            "asciiname": self.asciiname,
            "lat": self.lat,
            "lon": self.lon,
            "country": self.country,
            "admin1": self.admin1,
            "population": self.population,
        }


@dataclass(frozen=True)
class GeoMatch:
    place: GeoPlace
    score: float
    matched_on: str
    method: GeoMethod
    ambiguous: bool = False
    candidates: tuple[GeoPlace, ...] = field(default_factory=tuple)

    @property
    def confidence(self) -> float:
        """Composite confidence in [0, 1]."""
        method_prior = {
            GeoMethod.EXACT: 1.00,
            GeoMethod.REGION_TABLE: 0.95,
            GeoMethod.TRIGRAM_FUZZY: 0.90,
            GeoMethod.QUERY: 0.85,
            GeoMethod.REGEX: 0.70,
            GeoMethod.MANUAL: 1.00,
        }.get(self.method, 0.80)
        raw = (self.score / 100.0) * method_prior
        return raw * (0.85 if self.ambiguous else 1.0)


def _to_unit(lat: float, lon: float) -> np.ndarray:
    """Convert (lat, lon) in degrees to a 3D unit vector."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    return np.array([
        math.cos(phi) * math.cos(lam),
        math.cos(phi) * math.sin(lam),
        math.sin(phi),
    ])


def _radius_to_chord(radius_km: float) -> float:
    """Convert great-circle distance (km) to 3D chord length on unit sphere."""
    theta = radius_km / _EARTH_RADIUS_KM
    return 2.0 * math.sin(theta / 2.0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Exact great-circle distance in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def distance_decay_boost(
    dist_km: float,
    max_boost: float = _GEO_BOOST_MAX,
    half_km: float = _GEO_DECAY_HALF_KM,
) -> float:
    """Exponential decay: max_boost * 2^(-dist / half_km)."""
    return max_boost * math.exp(-math.log(2) * dist_km / half_km)


def compute_geo_boost(
    chunk_lat: float,
    chunk_lon: float,
    query_lat: float,
    query_lon: float,
    radius_km: float,
    geo_confidence: float = 1.0,
) -> float:
    """Confidence-scaled, distance-decaying additive score bonus."""
    dist = haversine_km(chunk_lat, chunk_lon, query_lat, query_lon)
    if dist > radius_km * 2.0:
        return 0.0
    return distance_decay_boost(dist, half_km=radius_km / 2.0) * min(1.0, geo_confidence)


def _file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()[:16]


def _check_stale(path: str) -> None:
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    if age_days > _STALE_DAYS:
        log.warning(
            "cities500.txt is %.0f days old (>%d). Re-download: https://download.geonames.org/export/dump/",
            age_days,
            _STALE_DAYS,
        )


def save_version_info(path: str, extra: dict | None = None) -> None:
    info = {
        "path": path,
        "checksum": _file_checksum(path),
        "mtime": os.path.getmtime(path),
        "size_mb": round(os.path.getsize(path) / 1e6, 1),
        "age_days": round((time.time() - os.path.getmtime(path)) / 86400, 1),
        "build_timestamp": time.time(),
        "attribution": "GeoNames (https://www.geonames.org) - CC BY 4.0",
        **(extra or {}),
    }
    Path(_VERSION_PATH).write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_version_info() -> dict | None:
    try:
        return json.loads(Path(_VERSION_PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _trigrams(s: str) -> set[str]:
    padded = f"  {s}  "
    return {padded[i:i + _NGRAM_SIZE] for i in range(len(padded) - _NGRAM_SIZE + 1)}


WeightedTerm = tuple[str, float]


class OfflineGeocoder:
    def __init__(self, geonames_path: str = _GEONAMES_PATH) -> None:
        self._path = geonames_path
        self._state = GeocoderState.COLD
        self._error: Optional[str] = None
        self._build_ts: Optional[float] = None
        self._ready = threading.Event()
        self._load_lock = threading.Lock()
        # KDTree is immutable after load; guarded for free-threaded runtimes.
        self._kdtree_lock = threading.Lock()

        self.places_by_id: dict[int, GeoPlace] = {}
        self.alias_to_ids: dict[str, list[int]] = {}
        self._id_to_aliases: dict[int, list[str]] = {}
        self._ngram_to_ids: dict[str, set[int]] = {}
        self.kdtree: Optional[cKDTree] = None
        self._idx_to_id: list[int] = []
        self._id_to_idx: dict[int, int] = {}
        self._place_count = 0
        self._alias_count = 0

    def warm(self, background: bool = False) -> None:
        if self._ready.is_set():
            return
        if background:
            thread = threading.Thread(target=self._load_safe, name="geocoder-warm", daemon=True)
            thread.start()
            return
        self._load_safe()

    def is_available(self) -> bool:
        return self._ready.is_set()

    def status(self) -> dict:
        version = load_version_info()
        return {
            "state": self._state.value,
            "error": self._error,
            "places_count": self._place_count,
            "alias_count": self._alias_count,
            "build_ts": self._build_ts,
            "version_info": version,
            "attribution": "GeoNames (https://www.geonames.org) - CC BY 4.0",
        }

    def _load_safe(self) -> None:
        with self._load_lock:
            if self._ready.is_set():
                return
            self._state = GeocoderState.WARMING
            try:
                self._load()
                self._state = GeocoderState.READY
                self._build_ts = time.time()
                self._ready.set()
                log.info(
                    "Geocoder ready: %d places, %d aliases.",
                    self._place_count,
                    self._alias_count,
                )
            except Exception as exc:
                self._state = GeocoderState.FAILED
                self._error = str(exc)
                log.error("GeoNames load failed - geocoding disabled: %s", exc)

    def _load(self) -> None:
        if not Path(self._path).exists():
            raise FileNotFoundError(
                f"GeoNames data not found at {self._path}.\n"
                "Run: curl -L https://download.geonames.org/export/dump/cities500.zip "
                "-o data/cities500.zip && unzip data/cities500.zip -d data/"
            )

        _check_stale(self._path)
        log.info("Loading GeoNames from %s ...", self._path)
        t0 = time.time()

        df = pd.read_csv(
            self._path,
            sep="\t",
            header=None,
            names=_COLS,
            low_memory=False,
            dtype={
                "geonameid": "Int64",
                "lat": float,
                "lon": float,
                "population": "Int64",
            },
        ).dropna(subset=["lat", "lon", "geonameid"])

        places_by_id: dict[int, GeoPlace] = {}
        alias_to_ids: dict[str, list[int]] = defaultdict(list)
        id_to_aliases: dict[int, list[str]] = defaultdict(list)
        ngram_to_ids: dict[str, set[int]] = defaultdict(set)

        for row in df.itertuples(index=False, name="GeoRow"):
            gid = int(row.geonameid)
            pop = int(row.population) if pd.notna(row.population) else 0

            raw: list[str] = []
            if pd.notna(row.altnames):
                raw = [a.strip() for a in str(row.altnames).split(",") if a.strip()]

            all_names = {str(row.name), str(row.asciiname)} | set(raw)
            top_aliases = tuple(sorted(all_names, key=len)[:10])

            places_by_id[gid] = GeoPlace(
                geonameid=gid,
                name=str(row.name),
                asciiname=str(row.asciiname),
                lat=float(row.lat),
                lon=float(row.lon),
                country=str(row.country),
                admin1=str(row.admin1),
                population=pop,
                top_aliases=top_aliases,
            )

            for name in all_names:
                original = name.strip()
                key = original.lower()
                if not key:
                    continue
                alias_to_ids[key].append(gid)
                if original not in id_to_aliases[gid]:
                    id_to_aliases[gid].append(original)
                for tg in _trigrams(key):
                    ngram_to_ids[tg].add(gid)

        for key in alias_to_ids:
            alias_to_ids[key].sort(
                key=lambda geonameid: places_by_id[geonameid].population,
                reverse=True,
            )

        ordered_ids = list(places_by_id.keys())
        unit_vecs = np.vstack([
            _to_unit(places_by_id[geonameid].lat, places_by_id[geonameid].lon)
            for geonameid in ordered_ids
        ])

        self.places_by_id = places_by_id
        self.alias_to_ids = dict(alias_to_ids)
        self._id_to_aliases = dict(id_to_aliases)
        self._ngram_to_ids = dict(ngram_to_ids)
        self.kdtree = cKDTree(unit_vecs)
        self._idx_to_id = ordered_ids
        self._id_to_idx = {geonameid: idx for idx, geonameid in enumerate(ordered_ids)}
        self._place_count = len(places_by_id)
        self._alias_count = sum(len(v) for v in alias_to_ids.values())

        log.info("GeoNames loaded in %.1fs.", time.time() - t0)
        try:
            save_version_info(
                self._path,
                {
                    "place_count": self._place_count,
                    "alias_count": self._alias_count,
                },
            )
        except Exception as exc:
            log.debug("Could not write geo version info: %s", exc)

    def _trigram_candidates(self, query: str) -> list[int]:
        score: dict[int, int] = defaultdict(int)
        for trigram in _trigrams(query):
            for gid in self._ngram_to_ids.get(trigram, set()):
                score[gid] += 1
        return sorted(score, key=score.__getitem__, reverse=True)[:_NGRAM_CANDIDATES]

    def _flat_aliases_for(self, gids: list[int]) -> tuple[list[str], list[int]]:
        aliases: list[str] = []
        gid_map: list[int] = []
        for gid in gids:
            place = self.places_by_id.get(gid)
            if place is None:
                continue
            for alias in place.top_aliases:
                aliases.append(alias.lower())
                gid_map.append(gid)
        return aliases, gid_map

    def _disambiguate(self, gids: list[int], context_words: tuple[str, ...]) -> int:
        """Pick best candidate geonameid using weighted context + population."""
        ctx_tokens = {w.lower() for w in context_words if len(w) >= 2 and w.isalpha()}
        ctx_country_codes = {
            _COUNTRY_HINT_TO_CODE[token]
            for token in ctx_tokens
            if token in _COUNTRY_HINT_TO_CODE
        }
        if not ctx_tokens:
            return max(gids, key=lambda gid: self.places_by_id[gid].population)

        def _feature_score(gid: int) -> tuple[float, int]:
            place = self.places_by_id[gid]
            score = 0.0

            if place.country.upper() in ctx_country_codes:
                score += 4.0
            elif place.country.lower() in ctx_tokens:
                score += 3.0

            admin1_tokens = set(place.admin1.lower().split())
            if admin1_tokens & ctx_tokens:
                score += 2.0

            aliases = self._id_to_aliases.get(gid, list(place.top_aliases))
            for alias in aliases[:40]:
                alias_tokens = {
                    token.lower()
                    for token in re.findall(r"[A-Za-z]{2,}", alias)
                }
                if alias_tokens & ctx_tokens:
                    score += 2.5
                    break

            pop_prior = math.log10(max(place.population, 1)) / 8.0
            score += pop_prior * 0.5
            return score, place.population

        return max(gids, key=_feature_score)

    @lru_cache(maxsize=_FORWARD_CACHE_SIZE)
    def forward(
        self,
        place_name: str,
        threshold: int = _DEFAULT_THRESHOLD,
        context_words: tuple[str, ...] = (),
    ) -> GeoMatch | None:
        """Forward geocode one place name with exact/region/fuzzy fallback."""
        if not self._ready.is_set():
            return None

        query = place_name.lower().strip()

        if query in self.alias_to_ids:
            gids = self.alias_to_ids[query]
            runner_up = self.places_by_id.get(gids[1]) if len(gids) > 1 else None
            primary = self.places_by_id[gids[0]]
            ambiguous = (
                runner_up is not None and runner_up.population > primary.population * 0.05
            )
            best_gid = (
                self._disambiguate(gids[:10], context_words)
                if ambiguous and context_words
                else gids[0]
            )
            return GeoMatch(
                place=self.places_by_id[best_gid],
                score=100.0,
                matched_on=query,
                method=GeoMethod.EXACT,
                ambiguous=ambiguous,
                candidates=tuple(self.places_by_id[gid] for gid in gids[:5]),
            )

        if query in NAMED_REGIONS:
            lat, lon, _ = NAMED_REGIONS[query]
            nearby = self.find_near(lat, lon, radius_km=50.0)
            if nearby:
                proxy = max(nearby, key=lambda place: place.population)
                return GeoMatch(
                    place=proxy,
                    score=95.0,
                    matched_on=query,
                    method=GeoMethod.REGION_TABLE,
                    ambiguous=False,
                )
            return None

        candidate_gids = self._trigram_candidates(query)
        if not candidate_gids:
            return None
        aliases, gid_map = self._flat_aliases_for(candidate_gids)
        if not aliases:
            return None

        result = process.extractOne(
            query,
            aliases,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is None:
            return None

        matched_alias, score, local_idx = result
        best_gid = gid_map[local_idx]

        alt_results = process.extract(
            query,
            aliases,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=score - _AMBIGUITY_WINDOW,
            limit=20,
        )
        alt_gids = list({gid_map[idx] for _, _, idx in alt_results if gid_map[idx] != best_gid})
        ambiguous = bool(alt_gids)

        if ambiguous and context_words:
            best_gid = self._disambiguate([best_gid] + alt_gids, context_words)

        return GeoMatch(
            place=self.places_by_id[best_gid],
            score=float(score),
            matched_on=matched_alias,
            method=GeoMethod.TRIGRAM_FUZZY,
            ambiguous=ambiguous,
            candidates=tuple(
                self.places_by_id[gid]
                for gid in sorted(
                    alt_gids,
                    key=lambda geonameid: self.places_by_id[geonameid].population,
                    reverse=True,
                )[:4]
            ),
        )

    def resolve_all(
        self,
        place_names: list[str],
        query: str = "",
        threshold: int = _DEFAULT_THRESHOLD,
    ) -> list[GeoMatch]:
        """Resolve all candidate place names from one query."""
        all_ctx = tuple({word for word in re.findall(r"[A-Z][a-zA-Z]{2,}", query)})
        results: list[GeoMatch] = []
        for name in place_names:
            ctx = tuple(word for word in all_ctx if word.lower() not in name.lower())
            match = self.forward(name, threshold=threshold, context_words=ctx)
            if match:
                results.append(match)
        return results

    def find_near(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        max_results: int = 200,
    ) -> list[GeoPlace]:
        """Return canonical places within radius_km."""
        if not self._ready.is_set() or self.kdtree is None:
            return []

        chord = _radius_to_chord(radius_km)
        qvec = _to_unit(lat, lon)
        with self._kdtree_lock:
            raw_idxs = self.kdtree.query_ball_point(qvec, chord)

        results: list[tuple[float, GeoPlace]] = []
        for idx in raw_idxs:
            gid = self._idx_to_id[idx]
            place = self.places_by_id[gid]
            dist = haversine_km(lat, lon, place.lat, place.lon)
            if dist <= radius_km:
                results.append((dist, place))

        results.sort(key=lambda pair: pair[0])
        return [place for _, place in results[:max_results]]

    def reverse(self, lat: float, lon: float, k: int = 1) -> list[tuple[float, GeoPlace]]:
        """Return nearest canonical places as (distance_km, GeoPlace)."""
        if not self._ready.is_set() or self.kdtree is None:
            return []

        qvec = _to_unit(lat, lon)
        with self._kdtree_lock:
            _, idxs = self.kdtree.query(qvec, k=k)

        if k == 1:
            idxs = [idxs]

        output: list[tuple[float, GeoPlace]] = []
        for idx in idxs:
            gid = self._idx_to_id[idx]
            place = self.places_by_id[gid]
            output.append((
                haversine_km(lat, lon, place.lat, place.lon),
                place,
            ))
        return output

    def get_aliases(self, geonameid: int) -> list[str]:
        """O(1) reverse alias lookup by geonameid."""
        return self._id_to_aliases.get(geonameid, [])

    def spatial_center(self, place_name: str) -> tuple[float, float, float] | None:
        """Return (lat, lon, radius_km) for a place name or named region."""
        query = place_name.lower().strip()
        if query in NAMED_REGIONS:
            return NAMED_REGIONS[query]
        match = self.forward(query)
        return (match.place.lat, match.place.lon, 50.0) if match else None


def extract_places_from_query(query: str) -> list[str]:
    """Extract all candidate place names from a query string."""
    found: list[str] = []
    seen_lower: set[str] = set()

    def _add(value: str) -> None:
        cleaned = value.strip()
        key = cleaned.lower()
        if key not in seen_lower:
            found.append(cleaned)
            seen_lower.add(key)

    for match in _QUOTED_PLACE.finditer(query):
        _add(match.group(1))

    for match in _SPATIAL_CUES.finditer(query):
        after = query[match.end():]
        chain_match = _CAP_NP_CHAIN.match(after)
        if chain_match:
            for part in re.split(r"\s+and\s+", chain_match.group(1), maxsplit=4):
                _add(part)
            continue

        np_match = _CAP_NP_STRICT.match(after)
        if np_match:
            _add(np_match.group(1))

    for match in _BETWEEN_CHAIN.finditer(query):
        _add(match.group(1))
        _add(match.group(2))

    lower_query = query.lower()
    for region in NAMED_REGIONS:
        if re.search(r"\b" + re.escape(region) + r"\b", lower_query):
            _add(region.title())

    return found


def build_geo_query_expansion(
    match: GeoMatch,
    geocoder: OfflineGeocoder,
    max_terms: int = _MAX_EXPANSION_TERMS,
) -> list[WeightedTerm]:
    """Build weighted expansion terms for BM25/FTS query augmentation."""
    canonical = match.place.name
    all_aliases = geocoder.get_aliases(match.place.geonameid)
    result: list[WeightedTerm] = [(canonical, 1.0)]
    seen: set[str] = {canonical.lower()}

    scored: list[tuple[str, float]] = []
    for raw_alias in all_aliases:
        alias = raw_alias.strip()
        if not alias or alias.lower() in seen:
            continue
        sim = fuzz.ratio(canonical.lower(), alias.lower()) / 100.0
        weight = 0.4 + 0.3 * sim
        scored.append((alias, round(weight, 2)))

    scored.sort(key=lambda item: item[1], reverse=True)
    for alias, weight in scored[:max_terms - 1]:
        result.append((alias, weight))
        seen.add(alias.lower())

    return result


_instance: Optional[OfflineGeocoder] = None
_instance_lock = threading.Lock()


def get_geocoder(path: str = _GEONAMES_PATH) -> OfflineGeocoder:
    """Return singleton geocoder instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = OfflineGeocoder(path)
    return _instance
