# -*- coding: utf-8 -*-
"""
region_3d.py
============
??"????????"?? DataFrame ???? 3D ???????????

?????
    - community_agg : ????????????lon, lat, ????, ??????, ?????
    - patient_layer : ?????????????????????????????? jitter????????????????

??????
    - ?????????? numpy + pandas??
    - ???????????? df????? dict-of-list??pydeck ?????
    - ????? RGBA [0-255]???? pydeck ???????
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.geo import all_community_geo, BEIJING_CENTER, get_community_geo


# ??????????RGBA 0-255??
RISK_COLOR = {
    "??":   [220, 50, 47, 220],     # ??
    "??":   [255, 165, 0, 220],     # ??
    "??":   [38, 139, 210, 220],    # ??
    "?????????????????": [180, 180, 180, 160],  # ??
}

DR_COLOR = {
    0: [38, 139, 210, 200],   # ??DR - ??
    1: [133, 199, 233, 200],  # ??? - ???
    2: [255, 220, 100, 200],  # ???? - ??
    3: [255, 140, 0, 200],    # ??? - ??
    4: [220, 50, 47, 220],    # ??? - ??
}


# ----------------------------- ??????? ----------------------------- #
def aggregate_by_community(df: pd.DataFrame, community_col: str = "community") -> pd.DataFrame:
    """
    ???????????????? = ?????????
    ?????community, lon, lat, base_height, n, mean_dr,
          high_risk_n, high_risk_pct, color, elevation
    """
    geo = all_community_geo()
    rows: List[Dict[str, Any]] = []

    if community_col not in df.columns or len(df) == 0:
        # ?????????????????
        for name, g in geo.items():
            rows.append(_empty_community_row(name, g))
        return pd.DataFrame(rows)

    for name, g in df.groupby(community_col):
        g_geo = geo.get(name) or get_community_geo(name)
        n = len(g)
        if n == 0:
            rows.append(_empty_community_row(name, g_geo))
            continue

        mean_dr = float(g["dr_grade"].mean()) if "dr_grade" in g.columns else 0.0
        # ?????????????????+????+DR ????????????
        cv_high = (g.get("cv_risk", pd.Series(dtype=object)) == "??").sum() if "cv_risk" in g.columns else 0
        htn_high = (g.get("htn_risk", pd.Series(dtype=object)) == "??").sum() if "htn_risk" in g.columns else 0
        high_risk_n = int(cv_high + htn_high + (g["dr_grade"] >= 3).sum() if "dr_grade" in g.columns else 0)
        high_risk_pct = (high_risk_n / max(n, 1))

        # ?????????????"?????"
        if high_risk_pct >= 0.20:
            color = RISK_COLOR["??"]
        elif high_risk_pct >= 0.08:
            color = RISK_COLOR["??"]
        else:
            color = RISK_COLOR["??"]

        # ???? = ??????? ?? ???????
        base_h = g_geo["base_height"]
        # ?????????1.0 ~ 2.5
        risk_factor = 1.0 + min(high_risk_pct * 5.0, 1.5) + min(mean_dr / 4.0, 0.5)
        elevation = base_h * risk_factor

        rows.append({
            "community": name,
            "lon": g_geo["lon"],
            "lat": g_geo["lat"],
            "base_height": base_h,
            "n": int(n),
            "mean_dr": round(mean_dr, 2),
            "high_risk_n": high_risk_n,
            "high_risk_pct": round(high_risk_pct, 4),
            "color": color,
            "elevation": float(elevation),
        })

    # ????????????????
    have = {r["community"] for r in rows}
    for name, g_geo in geo.items():
        if name not in have:
            rows.append(_empty_community_row(name, g_geo))

    return pd.DataFrame(rows).sort_values("community").reset_index(drop=True)


def _empty_community_row(name: str, g_geo: Dict[str, float]) -> Dict[str, Any]:
    """??????????????????????? 0?????????"""
    return {
        "community": name,
        "lon": g_geo["lon"],
        "lat": g_geo["lat"],
        "base_height": g_geo["base_height"],
        "n": 0,
        "mean_dr": 0.0,
        "high_risk_n": 0,
        "high_risk_pct": 0.0,
        "color": [200, 200, 200, 100],
        "elevation": 0.0,
    }


# ----------------------------- ???????jitter?? ----------------------------- #
def make_patient_points(
    df: pd.DataFrame,
    community_col: str = "community",
    max_per_community: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """
    ????????????????????????????? jitter??~0.005?? ?? 500m????
    ????"????????"??????????? / ??????
    """
    rng = np.random.default_rng(seed)
    geo = all_community_geo()
    pts: List[Dict[str, Any]] = []

    if len(df) == 0 or community_col not in df.columns:
        return pd.DataFrame(columns=["lon", "lat", "dr_grade", "cv_risk", "color"])

    for comm, g in df.groupby(community_col):
        g_geo = geo.get(comm)
        if g_geo is None:
            continue
        # ????????????
        if len(g) > max_per_community:
            g = g.sample(n=max_per_community, random_state=seed)
        for _, r in g.iterrows():
            # ??????????? ??0.005?? ????
            lon = g_geo["lon"] + float(rng.normal(0, 0.003))
            lat = g_geo["lat"] + float(rng.normal(0, 0.003))
            grade = int(r.get("dr_grade", 0))
            color = DR_COLOR.get(grade, [180, 180, 180, 180])
            pts.append({
                "lon": lon,
                "lat": lat,
                "dr_grade": grade,
                "cv_risk": str(r.get("cv_risk", "??")),
                "color": color,
            })
    return pd.DataFrame(pts)


# ----------------------------- ??????? ----------------------------- #
def get_deck_view_state(zoom: float = 9.5, pitch: float = 50.0, bearing: float = -20.0) -> Dict[str, float]:
    """pydeck ???????????? + ???? 50???? 3D ???????????"""
    return {
        "longitude": BEIJING_CENTER[0],
        "latitude": BEIJING_CENTER[1],
        "zoom": zoom,
        "pitch": pitch,
        "bearing": bearing,
    }


def make_building_layer(agg_df: pd.DataFrame) -> Dict[str, Any]:
    """
    ??????? Layer ?? spec ???
    ?????????? dict ???? pydeck.Layer??
    ?????????????? pydeck.Layer ???? dict??????????? streamlit ???????
    """
    return {
        "@@type": "ColumnLayer",
        "data": agg_df.to_dict("records"),
        "diskResolution": 24,
        "radius": 600,
        "extruded": True,
        "pickable": True,
        "elevationScale": 1.0,
        "getPosition": "[lon, lat]",
        "getElevation": "elevation",
        "getFillColor": "color",
        "getLineColor": [255, 255, 255, 80],
        "lineWidthMinPixels": 1,
    }


def make_heatmap_layer(pts_df: pd.DataFrame) -> Dict[str, Any]:
    """??????? spec???????????????????"""
    return {
        "@@type": "HeatmapLayer",
        "data": pts_df.to_dict("records"),
        "getPosition": "[lon, lat]",
        "radiusPixels": 60,
        "intensity": 1,
        "threshold": 0.05,
    }


def make_patient_scatter_layer(pts_df: pd.DataFrame) -> Dict[str, Any]:
    """??????? spec??"""
    return {
        "@@type": "ScatterplotLayer",
        "data": pts_df.to_dict("records"),
        "getPosition": "[lon, lat]",
        "getFillColor": "color",
        "getRadius": 80,
        "radiusMinPixels": 3,
        "radiusMaxPixels": 8,
        "pickable": True,
        "opacity": 0.7,
    }
