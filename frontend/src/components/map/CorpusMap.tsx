"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Globe, Trash2 } from "lucide-react";
import { Map, Layer, Popup, Source } from "@vis.gl/react-maplibre";
import type { MapRef, MapLayerMouseEvent } from "@vis.gl/react-maplibre";
import type { Feature, FeatureCollection, Point } from "geojson";
import { layers as protomapsLayers, namedFlavor } from "@protomaps/basemaps";
import { Protocol } from "pmtiles";
import type { GeoJSONSource, LayerSpecification, StyleSpecification } from "maplibre-gl";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import {
  sourceApi,
  type GeoMentionDetail,
  type GeoMentionGroup,
  type GeoMentionsResponse,
} from "@/lib/api-client";
import { getBackendApiBase } from "@/lib/backend-url";
import { useAppDispatch } from "@/context/app-context";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

type GeoFeatureProperties = {
  place_name: string;
  geonameid: number;
  mention_count: number;
  max_confidence: number;
  source_ids: string[];
  chunk_ids: string[];
  matched_inputs: string[];
  mention_ids: string[];
  mentions: GeoMentionDetail[];
};

type HoverPopup = {
  lng: number;
  lat: number;
  placeName: string;
  mentionCount: number;
};

const CLUSTER_LAYER: LayerSpecification = {
  id: "clusters",
  source: "geo-mentions",
  type: "circle",
  filter: ["has", "point_count"],
  paint: {
    "circle-color": ["step", ["get", "point_count"], "#3b82f6", 5, "#1d4ed8", 20, "#1e3a8a"],
    "circle-radius": ["step", ["get", "point_count"], 18, 5, 26, 20, 34],
    "circle-opacity": 0.85,
  },
};

const CLUSTER_COUNT_LAYER: LayerSpecification = {
  id: "cluster-count",
  source: "geo-mentions",
  type: "symbol",
  filter: ["has", "point_count"],
  layout: {
    "text-field": "{point_count_abbreviated}",
    "text-font": ["Noto Sans Regular"],
    "text-size": 12,
  },
  paint: {
    "text-color": "#ffffff",
  },
};

let pmtilesProtocol: Protocol | null = null;
let pmtilesProtocolRegistered = false;

const OFFLINE_TILE_MAX_ZOOM = (() => {
  const parsed = Number.parseInt(process.env.NEXT_PUBLIC_BASEMAP_MAX_ZOOM ?? "7", 10);
  if (parsed === 6 || parsed === 7) {
    return parsed;
  }
  return 7;
})();

const OFFLINE_TILE_MIN_ZOOM = 1;
const PMTILES_SOURCE_PATH = process.env.NEXT_PUBLIC_PM_TILES_PATH ?? "/basemap/world_cities.pmtiles";
const PMTILES_GLYPHS_PATH =
  process.env.NEXT_PUBLIC_PM_TILES_GLYPHS ?? "/basemap-assets/fonts/{fontstack}/{range}.pbf";
const DEFAULT_PM_TILES_SOURCE_URL = "https://demo-bucket.protomaps.com/v4.pmtiles";
const BASEMAP_STATUS_POLL_MS = 1200;

const CARTO_DARK_FLAVOR = {
  ...namedFlavor("black"),
  background: "#0e0e0e",
  earth: "#1a1c26",
  water: "#242b35",
  water_shadow: "#1e252e",
  boundaries: "#3d3f4d",
  country_label: "#c0c0d0",
  state_label: "#8a8aa0",
  state_label_halo: "#1a1c26",
  city_label: "#b0b0c8",
  city_label_halo: "#1a1c26",
  ocean_label: "#7a9aaa",
  subplace_label: "#8a8aa0",
  subplace_label_halo: "#1a1c26",
};

const INCLUDE_BASEMAP_LAYER = /(country|boundary|admin|coast|coastline|ocean|water|city|town|place)/i;
const EXCLUDE_BASEMAP_LAYER = /(road|street|path|rail|transit|building|poi|airport|aeroway|runway|bridge|tunnel)/i;

type BasemapSetupStatus = {
  status: "idle" | "running" | "ready" | "error";
  progress: number;
  message: string;
  error?: string | null;
  file_exists?: boolean;
};

function resolveAbsoluteUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) {
    return pathOrUrl;
  }
  if (typeof window === "undefined") {
    return pathOrUrl;
  }
  const normalized = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${window.location.origin}${normalized}`;
}

async function urlExists(pathOrUrl: string): Promise<boolean> {
  const target = resolveAbsoluteUrl(pathOrUrl);
  try {
    const head = await fetch(target, { method: "HEAD", cache: "no-store" });
    if (head.ok) {
      return true;
    }
    if (head.status !== 405) {
      return false;
    }

    // Some static hosts disallow HEAD; probe with a tiny byte range instead.
    const ranged = await fetch(target, {
      method: "GET",
      cache: "no-store",
      headers: { Range: "bytes=0-0" },
    });
    return ranged.ok || ranged.status === 206;
  } catch {
    return false;
  }
}

function ensurePmtilesProtocolRegistered() {
  if (pmtilesProtocolRegistered) {
    return;
  }
  pmtilesProtocol = new Protocol({ metadata: true });
  maplibregl.addProtocol("pmtiles", pmtilesProtocol.tile);
  pmtilesProtocolRegistered = true;
}

function buildOfflineBasemapStyle(opts: {
  pmtilesUrl: string;
}): StyleSpecification {
  const basemapSourceName = "protomaps-city";
  const layers = protomapsLayers(basemapSourceName, CARTO_DARK_FLAVOR, {
    lang: "en",
  }) as LayerSpecification[];

  const filteredLayers = layers
    .filter((layer) => {
      const id = String(layer.id ?? "");
      if (!id) {
        return false;
      }
      if (EXCLUDE_BASEMAP_LAYER.test(id)) {
        return false;
      }
      return INCLUDE_BASEMAP_LAYER.test(id);
    })
    .map((layer) => {
      const capped: LayerSpecification = {
        ...layer,
        minzoom: layer.minzoom == null ? OFFLINE_TILE_MIN_ZOOM : Math.max(OFFLINE_TILE_MIN_ZOOM, layer.minzoom),
        maxzoom:
          layer.maxzoom == null ? OFFLINE_TILE_MAX_ZOOM : Math.min(OFFLINE_TILE_MAX_ZOOM, layer.maxzoom),
      };
      return capped;
    });

  return {
    version: 8,
    name: "corpus-offline-map",
    glyphs: PMTILES_GLYPHS_PATH,
    sources: {
      [basemapSourceName]: {
        type: "vector",
        url: `pmtiles://${opts.pmtilesUrl}`,
        minzoom: OFFLINE_TILE_MIN_ZOOM,
        maxzoom: OFFLINE_TILE_MAX_ZOOM,
        attribution: "© OpenStreetMap contributors",
      },
    },
    layers: [
      {
        id: "background",
        type: "background",
        paint: {
          "background-color": "#0e0e0e",
        },
      },
      ...filteredLayers,
    ],
  };
}

const OFFLINE_MAP_STYLE: StyleSpecification = {
  version: 8,
  name: "corpus-offline-map",
  sources: {},
  layers: [
    {
      id: "background",
      type: "background",
      paint: {
        "background-color": "#0e0e0e",
      },
    },
  ],
};

interface CorpusMapProps {
  onCountChange?: (count: number) => void;
  active?: boolean;
  refreshNonce?: number;
  threshold: number;
  sourceIds: string[];
}

function filterMentionsBySources(
  payload: GeoMentionsResponse,
  sourceIds: string[],
): GeoMentionsResponse {
  if (sourceIds.length === 0) {
    return { count: 0, mentions: [] };
  }

  const allowed = new Set(sourceIds);
  const mentions: GeoMentionGroup[] = [];

  for (const group of payload.mentions) {
    const details = (group.mentions ?? []).filter((mention) => allowed.has(mention.source_id));
    if (details.length === 0) {
      continue;
    }

    const sourceSet = new Set<string>();
    const chunkSet = new Set<string>();
    const matchedSet = new Set<string>();
    const mentionIds: string[] = [];
    let maxConfidence = 0;

    for (const detail of details) {
      sourceSet.add(detail.source_id);
      chunkSet.add(detail.chunk_id);
      matchedSet.add(detail.matched_input);
      mentionIds.push(detail.id);
      maxConfidence = Math.max(maxConfidence, detail.confidence);
    }

    mentions.push({
      ...group,
      mention_count: details.length,
      max_confidence: maxConfidence,
      source_ids: Array.from(sourceSet),
      chunk_ids: Array.from(chunkSet),
      matched_inputs: Array.from(matchedSet),
      mention_ids: mentionIds,
      mentions: details,
    });
  }

  return {
    count: mentions.length,
    mentions,
  };
}

export function CorpusMap({
  onCountChange,
  active = true,
  refreshNonce = 0,
  threshold,
  sourceIds,
}: CorpusMapProps) {
  const dispatch = useAppDispatch();
  const queryClient = useQueryClient();
  const mapRef = useRef<MapRef | null>(null);
  const hasFitBounds = useRef(false);

  const [actionError, setActionError] = useState<string | null>(null);
  const [hoverPopup, setHoverPopup] = useState<HoverPopup | null>(null);
  const [selectedGroup, setSelectedGroup] = useState<GeoMentionGroup | null>(null);
  const [deletingMentionId, setDeletingMentionId] = useState<string | null>(null);
  const [pendingDeleteMention, setPendingDeleteMention] = useState<GeoMentionDetail | null>(null);
  const [mapStyle, setMapStyle] = useState<StyleSpecification>(OFFLINE_MAP_STYLE);
  const [basemapReady, setBasemapReady] = useState(false);
  const [basemapStatus, setBasemapStatus] = useState<BasemapSetupStatus | null>(null);

  const fetchBasemapStatus = useCallback(async (): Promise<BasemapSetupStatus> => {
    const res = await fetch(`${getBackendApiBase()}/basemap/setup/status`, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`Failed to check basemap status (HTTP ${res.status})`);
    }
    return (await res.json()) as BasemapSetupStatus;
  }, []);

  const startBasemapSetup = useCallback(async (): Promise<BasemapSetupStatus> => {
    const params = new URLSearchParams({
      max_zoom: String(OFFLINE_TILE_MAX_ZOOM),
      source_url: DEFAULT_PM_TILES_SOURCE_URL,
    });
    const res = await fetch(`${getBackendApiBase()}/basemap/setup?${params.toString()}`, {
      method: "POST",
    });
    if (!res.ok) {
      throw new Error(`Failed to start basemap setup (HTTP ${res.status})`);
    }
    return (await res.json()) as BasemapSetupStatus;
  }, []);

  const applyLocalBasemapStyle = useCallback(() => {
    setMapStyle(
      buildOfflineBasemapStyle({
        pmtilesUrl: resolveAbsoluteUrl(PMTILES_SOURCE_PATH),
      }),
    );
    setBasemapReady(true);
  }, []);

  const handleStartSetup = useCallback(async () => {
    try {
      const status = await startBasemapSetup();
      setBasemapStatus(status);
    } catch (err) {
      setBasemapStatus({
        status: "error",
        progress: 0,
        message: "Basemap setup failed to start.",
        error: err instanceof Error ? err.message : "Unknown setup error",
      });
    }
  }, [startBasemapSetup]);

  useEffect(() => {
    let cancelled = false;

    const initBasemap = async () => {
      try {
        const hasLocalPmtiles = await urlExists(PMTILES_SOURCE_PATH);
        if (cancelled) return;

        if (hasLocalPmtiles) {
          applyLocalBasemapStyle();
          setBasemapStatus({
            status: "ready",
            progress: 100,
            message: "Offline basemap is ready.",
            file_exists: true,
          });
          return;
        }

        const status = await fetchBasemapStatus();
        if (cancelled) return;
        setBasemapStatus(status);
        if (status.status === "ready" || status.file_exists) {
          applyLocalBasemapStyle();
        }
      } catch {
        setMapStyle(OFFLINE_MAP_STYLE);
        setBasemapReady(false);
        setBasemapStatus({
          status: "error",
          progress: 0,
          message: "Unable to initialize basemap.",
          error: "Failed to check local basemap status.",
        });
      }
    };

    try {
      ensurePmtilesProtocolRegistered();
    } catch {
      // Keep marker rendering available even if basemap protocol registration fails.
    }

    void initBasemap();

    return () => {
      cancelled = true;
    };
  }, [applyLocalBasemapStyle, fetchBasemapStatus]);

  useEffect(() => {
    if (basemapReady) {
      return;
    }
    if (!basemapStatus || basemapStatus.status !== "running") {
      return;
    }

    let cancelled = false;
    const interval = window.setInterval(async () => {
      try {
        const status = await fetchBasemapStatus();
        if (cancelled) return;
        setBasemapStatus(status);

        if (status.status === "ready" || status.file_exists) {
          applyLocalBasemapStyle();
        }
      } catch {
        if (cancelled) return;
        setBasemapStatus((prev) => ({
          status: "error",
          progress: prev?.progress ?? 0,
          message: "Basemap setup status check failed.",
          error: "Could not reach backend setup status endpoint.",
        }));
      }
    }, BASEMAP_STATUS_POLL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [applyLocalBasemapStyle, basemapReady, basemapStatus, fetchBasemapStatus]);

  const normalizedSourceIds = useMemo(() => {
    const seen = new Set<string>();
    const ids: string[] = [];
    for (const raw of sourceIds) {
      const sid = String(raw).trim();
      if (!sid || seen.has(sid)) continue;
      seen.add(sid);
      ids.push(sid);
    }
    return ids;
  }, [sourceIds]);

  const queryKey = useMemo(
    () => ["geo-mentions", Number(threshold.toFixed(2)), normalizedSourceIds, refreshNonce] as const,
    [threshold, normalizedSourceIds, refreshNonce],
  );

  const fetchMentions = useCallback(async (): Promise<GeoMentionsResponse> => {
    return sourceApi.getGeoMentions(undefined, threshold, 1000, 0, true, normalizedSourceIds);
  }, [threshold, normalizedSourceIds]);

  const mentionsQuery = useQuery<GeoMentionsResponse>({
    queryKey,
    queryFn: fetchMentions,
    enabled: active && normalizedSourceIds.length > 0,
  });

  const data = useMemo(() => {
    const payload = mentionsQuery.data ?? { count: 0, mentions: [] };
    return filterMentionsBySources(payload, normalizedSourceIds);
  }, [mentionsQuery.data, normalizedSourceIds]);
  const isLoading = mentionsQuery.isLoading || mentionsQuery.isFetching;
  const error = actionError ?? (mentionsQuery.error instanceof Error ? mentionsQuery.error.message : null);

  const confidenceHighStop = useMemo(() => Math.min(1.0, threshold + 0.17), [threshold]);
  const unclusteredLayer = useMemo<LayerSpecification>(
    () => ({
      id: "unclustered",
      source: "geo-mentions",
      type: "circle",
      filter: ["!", ["has", "point_count"]],
      paint: {
        "circle-color": ["interpolate", ["linear"], ["get", "max_confidence"], threshold, "#f59e0b", confidenceHighStop, "#3b82f6"],
        "circle-opacity": ["interpolate", ["linear"], ["get", "max_confidence"], threshold, 0.45, confidenceHighStop, 1.0],
        "circle-radius": 7,
        "circle-stroke-width": 1,
        "circle-stroke-color": "#ffffff",
      },
    }),
    [confidenceHighStop, threshold],
  );

  useEffect(() => {
    setActionError(null);
  }, [queryKey]);

  useEffect(() => {
    if (!active || normalizedSourceIds.length === 0) {
      setHoverPopup(null);
      setSelectedGroup(null);
    }
  }, [active, normalizedSourceIds.length]);

  useEffect(() => {
    onCountChange?.(data.count);
  }, [data.count, onCountChange]);

  const geojson = useMemo<FeatureCollection<Point, GeoFeatureProperties>>(() => {
    const features: Array<Feature<Point, GeoFeatureProperties>> = data.mentions.map((group: GeoMentionGroup) => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [group.lon, group.lat],
      },
      properties: {
        place_name: group.place_name,
        geonameid: group.geonameid,
        mention_count: group.mention_count,
        max_confidence: group.max_confidence,
        source_ids: group.source_ids,
        chunk_ids: group.chunk_ids,
        matched_inputs: group.matched_inputs,
        mention_ids: group.mention_ids,
        mentions: group.mentions,
      },
    }));

    return {
      type: "FeatureCollection",
      features,
    };
  }, [data.mentions]);

  useEffect(() => {
    if (!selectedGroup) {
      return;
    }
    const next = data.mentions.find((item: GeoMentionGroup) => item.geonameid === selectedGroup.geonameid) ?? null;
    if (next === selectedGroup) {
      return;
    }
    setSelectedGroup(next);
  }, [data.mentions, selectedGroup]);

  useEffect(() => {
    hasFitBounds.current = false;
  }, [geojson.features]);

  useEffect(() => {
    if (hasFitBounds.current) {
      return;
    }
    if (!geojson.features.length) {
      return;
    }

    const map = mapRef.current?.getMap();
    if (!map) {
      return;
    }

    let minLon = Infinity;
    let minLat = Infinity;
    let maxLon = -Infinity;
    let maxLat = -Infinity;

    for (const feature of geojson.features) {
      const [lon, lat] = feature.geometry.coordinates;
      minLon = Math.min(minLon, lon);
      minLat = Math.min(minLat, lat);
      maxLon = Math.max(maxLon, lon);
      maxLat = Math.max(maxLat, lat);
    }

    if (!Number.isFinite(minLon) || !Number.isFinite(minLat) || !Number.isFinite(maxLon) || !Number.isFinite(maxLat)) {
      return;
    }

    map.fitBounds(
      [
        [minLon, minLat],
        [maxLon, maxLat],
      ],
      { padding: 60, duration: 900 },
    );
    hasFitBounds.current = true;
  }, [geojson.features]);

  const onMapClick = useCallback((evt: MapLayerMouseEvent) => {
    const first = evt.features?.[0];
    if (!first) {
      return;
    }

    const map = mapRef.current?.getMap();
    if (!map) {
      return;
    }

    const properties = (first.properties ?? {}) as Record<string, unknown>;
    const geometry = first.geometry;

    if (properties.point_count != null) {
      const clusterId = Number(properties.cluster_id);
      const source = map.getSource("geo-mentions") as GeoJSONSource | undefined;
      if (!source || Number.isNaN(clusterId)) {
        return;
      }

      source
        .getClusterExpansionZoom(clusterId)
        .then((zoom) => {
          if (!geometry || geometry.type !== "Point") {
            return;
          }
          map.easeTo({
            center: geometry.coordinates as [number, number],
            zoom,
            duration: 450,
          });
        })
        .catch(() => {
          // ignore cluster zoom errors
        });
      return;
    }

    const geonameid = Number(properties.geonameid);
    if (Number.isNaN(geonameid)) {
      return;
    }

    const group = data.mentions.find((item: GeoMentionGroup) => item.geonameid === geonameid) ?? null;
    setSelectedGroup(group);
  }, [data.mentions]);

  const onMapMouseMove = useCallback((evt: MapLayerMouseEvent) => {
    const first = evt.features?.[0];
    if (!first) {
      setHoverPopup(null);
      return;
    }

    const properties = (first.properties ?? {}) as Record<string, unknown>;
    if (properties.point_count != null) {
      setHoverPopup(null);
      return;
    }

    if (!first.geometry || first.geometry.type !== "Point") {
      setHoverPopup(null);
      return;
    }

    setHoverPopup({
      lng: first.geometry.coordinates[0],
      lat: first.geometry.coordinates[1],
      placeName: String(properties.place_name ?? ""),
      mentionCount: Number(properties.mention_count ?? 0),
    });
  }, []);

  const handleViewChunk = useCallback(async (sourceId: string, chunkId: string, matchedInput: string) => {
    const chunk = await sourceApi.getChunk(sourceId, chunkId);
    dispatch({
      type: "SET_ACTIVE_CITATION",
      citation: {
        number: 0,
        source_id: sourceId,
        chunk_id: chunkId,
        page: chunk.page_number ?? null,
        header_path: chunk.header_path,
        chunk_text: chunk.chunk_text,
        highlight_text: matchedInput,
        highlight_scope_text: chunk.chunk_text,
      },
    });
  }, [dispatch]);

  const handleDeleteMention = useCallback(async (mentionId: string) => {
    if (!selectedGroup) {
      return;
    }

    setDeletingMentionId(mentionId);
    setActionError(null);
    try {
      await sourceApi.deleteGeoMention(mentionId);
      await queryClient.invalidateQueries({ queryKey, exact: true });
      const refreshedRaw = await queryClient.fetchQuery({
        queryKey,
        queryFn: fetchMentions,
        staleTime: 0,
      });
      const refreshed = filterMentionsBySources(refreshedRaw, normalizedSourceIds);
      const updatedGroup = refreshed.mentions.find((m: GeoMentionGroup) => m.geonameid === selectedGroup.geonameid) ?? null;
      setSelectedGroup(updatedGroup);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to delete geo mention");
    } finally {
      setDeletingMentionId(null);
    }
  }, [fetchMentions, normalizedSourceIds, queryClient, queryKey, selectedGroup]);

  const handleConfirmDeleteMention = useCallback(async () => {
    if (!pendingDeleteMention) return;
    await handleDeleteMention(pendingDeleteMention.id);
    setPendingDeleteMention(null);
  }, [handleDeleteMention, pendingDeleteMention]);

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-gray-400">
        Loading map...
      </div>
    );
  }

  if (!basemapReady) {
    const status = basemapStatus;
    const progress = Math.max(0, Math.min(100, Math.round(status?.progress ?? 0)));
    const isRunning = status?.status === "running";
    const isError = status?.status === "error";
    const canStart = !isRunning;

    return (
      <div className="relative h-full w-full overflow-hidden bg-[#0e0e0e]">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(90%_70%_at_50%_0%,rgba(96,165,250,0.18),rgba(10,15,26,0)_70%)]" />
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <div className="w-full max-w-lg rounded-2xl border border-white/12 bg-black/45 p-6 shadow-[0_20px_60px_rgba(0,0,0,0.45)] backdrop-blur-xl">
            <p className="text-sm font-semibold tracking-wide text-white/90">Offline Basemap Setup</p>
            <p className="mt-2 text-sm text-white/70">Set up the local map once, then use it offline.</p>

            <div className="mt-4 rounded-lg border border-white/12 bg-black/35 p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-white/75">
                <span>{status?.message ?? "Ready to set up basemap"}</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-sky-400 to-blue-500 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {isError && (
              <p className="mt-3 text-xs text-red-300">{status?.error ?? "Basemap setup failed."}</p>
            )}

            <div className="mt-5 flex items-center gap-3">
              <button
                type="button"
                onClick={() => void handleStartSetup()}
                disabled={!canStart}
                className="rounded-md border border-blue-300/40 bg-blue-500/20 px-4 py-2 text-sm font-medium text-blue-100 transition-colors hover:bg-blue-500/30 disabled:cursor-not-allowed disabled:opacity-55"
              >
                {isRunning ? "Setting Up..." : "Set Up Basemap"}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-sm text-red-300">
        {error}
      </div>
    );
  }

  if (normalizedSourceIds.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-gray-300">
        <Globe className="h-9 w-9 text-gray-500" />
        <p>No sources selected - choose at least one source to render map markers</p>
      </div>
    );
  }

  if (!data.count) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-gray-300">
        <Globe className="h-9 w-9 text-gray-500" />
        <p>No locations indexed yet - re-ingest a document with geotagging enabled</p>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#0e0e0e]">
      <Map
        ref={mapRef}
        initialViewState={{ longitude: 12, latitude: 25, zoom: 1.7 }}
        mapStyle={mapStyle}
        minZoom={OFFLINE_TILE_MIN_ZOOM}
        maxZoom={OFFLINE_TILE_MAX_ZOOM}
        interactiveLayerIds={["clusters", "unclustered"]}
        onClick={onMapClick}
        onMouseMove={onMapMouseMove}
      >
        <Source
          id="geo-mentions"
          type="geojson"
          data={geojson}
          cluster={true}
          clusterMaxZoom={Math.max(OFFLINE_TILE_MIN_ZOOM, OFFLINE_TILE_MAX_ZOOM - 1)}
          clusterRadius={50}
        >
          <Layer {...CLUSTER_LAYER} />
          <Layer {...CLUSTER_COUNT_LAYER} />
          <Layer {...unclusteredLayer} />
        </Source>

        {hoverPopup && (
          <Popup
            longitude={hoverPopup.lng}
            latitude={hoverPopup.lat}
            closeButton={false}
            closeOnClick={false}
            anchor="top"
            offset={10}
            className="pointer-events-none corpus-map-popup"
          >
            <div className="rounded-lg border border-white/15 bg-black/80 px-3 py-2 text-xs shadow-[0_10px_24px_rgba(0,0,0,0.45)] backdrop-blur-md">
              <p className="font-semibold text-white">{hoverPopup.placeName}</p>
              <p className="text-white/70">
                {hoverPopup.mentionCount} mention{hoverPopup.mentionCount === 1 ? "" : "s"}
              </p>
            </div>
          </Popup>
        )}
      </Map>

      <aside
        className={`absolute inset-y-0 right-0 z-30 w-[min(95%,26rem)] transform border-l border-white/12 bg-black/62 shadow-[-18px_0_40px_rgba(0,0,0,0.48)] backdrop-blur-xl transition-transform duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] ${
          selectedGroup ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.05)_0%,rgba(255,255,255,0.00)_22%)]" />

        <div className="relative z-10 flex items-center justify-between border-b border-white/10 px-4 py-3">
          <div>
            <p className="text-sm font-semibold tracking-tight text-white/92">{selectedGroup?.place_name ?? ""}</p>
            <p className="text-xs text-white/58">
              {selectedGroup?.mention_count ?? 0} mention{(selectedGroup?.mention_count ?? 0) === 1 ? "" : "s"}
            </p>
          </div>
          <button
            className="rounded-md border border-white/14 bg-white/5 px-2.5 py-1 text-xs text-white/74 transition-colors hover:bg-white/12 hover:text-white"
            onClick={() => setSelectedGroup(null)}
            aria-label="Close place details"
          >
            Close
          </button>
        </div>

        <div className="relative z-10 h-[calc(100%-56px)] overflow-y-auto px-3 py-3">
          {(selectedGroup?.mentions ?? []).map((mention) => (
            <div key={mention.id} className="mb-3 rounded-xl border border-white/12 bg-white/[0.035] p-3.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="truncate text-xs font-medium tracking-wide text-white/86">{mention.source_id}</p>
                <span className="rounded-md border border-white/18 bg-white/8 px-2 py-0.5 text-[10px] font-medium text-white/82">
                  {(mention.confidence * 100).toFixed(0)}%
                </span>
              </div>

              <p className="mb-3 text-xs text-white/68">
                mentioned as &apos;{mention.matched_input}&apos;
              </p>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => handleViewChunk(mention.source_id, mention.chunk_id, mention.matched_input)}
                  className="rounded-md border border-white/16 bg-white/[0.03] px-2.5 py-1 text-xs text-white/82 transition-colors hover:bg-white/10"
                >
                  View source
                </button>
                <button
                  type="button"
                  onClick={() => setPendingDeleteMention(mention)}
                  disabled={deletingMentionId === mention.id}
                  className="rounded-md border border-red-400/35 bg-red-500/[0.08] p-1 text-red-300 transition-colors hover:bg-red-500/[0.16] disabled:opacity-50"
                  title="Delete mention"
                  aria-label={`Delete mention ${mention.matched_input} from ${mention.source_id}`}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <Dialog
        open={pendingDeleteMention !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteMention(null);
        }}
      >
        <DialogContent className="max-w-md border border-white/18 bg-[#111317] text-white sm:rounded-xl [&>button]:border [&>button]:border-white/20 [&>button]:bg-black/45 [&>button]:text-white/80 [&>button]:opacity-100 [&>button]:hover:text-white">
          <DialogHeader>
            <DialogTitle className="text-base text-white">Delete Geo Mention?</DialogTitle>
            <DialogDescription className="text-xs text-white/65">
              This will permanently remove this map mention from the indexed geo annotations.
            </DialogDescription>
          </DialogHeader>

          <div className="rounded-md border border-red-400/25 bg-red-500/10 px-3 py-2 text-xs text-red-100">
            <span className="font-semibold">{pendingDeleteMention?.matched_input ?? "Mention"}</span>
            <span className="text-red-200/75"> in {pendingDeleteMention?.source_id ?? "source"}</span>
          </div>

          <DialogFooter className="flex items-center justify-between gap-2 sm:justify-between">
            <button
              type="button"
              onClick={() => setPendingDeleteMention(null)}
              className="rounded-md border border-white/18 bg-white/[0.04] px-3 py-1.5 text-xs text-white/82 transition-colors hover:bg-white/10"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => void handleConfirmDeleteMention()}
              disabled={deletingMentionId !== null}
              className="rounded-md border border-red-400/35 bg-red-500/[0.14] px-3 py-1.5 text-xs font-medium text-red-100 transition-colors hover:bg-red-500/[0.22] disabled:opacity-60"
            >
              Delete Mention
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
