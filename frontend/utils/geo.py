# -*- coding: utf-8 -*-
"""
geo.py
======
城市级地图的地理常量层。

注意：
    - 北京 10 个社区的中心点采用"近似真实坐标"（北京二环-五环分布），
      用于数字孪生大屏的 3D 可视化，**非真实地理位置**，仅作科研演示。
    - 稳定随机种子：保证每次运行社区相对位置一致，方便答辩/录屏。
    - 提供：
        * COMMUNITY_LONGLAT  : {社区名: (lon, lat)}
        * BEIJING_CENTER     : (lon, lat)
        * get_community_geo(name) -> dict(lon, lat, ...)
"""

from __future__ import annotations

from typing import Dict, Tuple

# 北京天安门附近（地图中心）
BEIJING_CENTER: Tuple[float, float] = (116.4074, 39.9042)

# 10 个北京社区的近似经纬度（lon, lat）
# 分布：二环内（东城/西城）→ 五环外（昌平/大兴），均匀铺开
COMMUNITY_LONGLAT: Dict[str, Tuple[float, float]] = {
    "东城社区": (116.4180, 39.9280),   # 二环内，东侧
    "西城社区": (116.3660, 39.9120),   # 二环内，西侧
    "和平社区": (116.4080, 39.9180),   # 二环内，中部
    "海淀社区": (116.3100, 39.9840),   # 西北三-四环
    "朝阳社区": (116.4900, 39.9300),   # 东部三-四环
    "丰台社区": (116.2870, 39.8580),   # 南部三环
    "石景山社区": (116.2220, 39.9060), # 西部
    "通州社区": (116.6570, 39.9090),   # 东部副中心
    "昌平社区": (116.2350, 40.2200),   # 北部
    "大兴社区": (116.3410, 39.7280),   # 南部
}

# 社区在地图上的"健康指数"基准（用于 3D 柱高度，仅演示）
# 数值越大 → 柱越高 → 风险越高，越显眼的"红色"
BASE_HEIGHT_BY_COMMUNITY: Dict[str, float] = {
    "东城社区": 320.0,
    "西城社区": 280.0,
    "和平社区": 360.0,
    "海淀社区": 420.0,
    "朝阳社区": 480.0,
    "丰台社区": 380.0,
    "石景山社区": 260.0,
    "通州社区": 300.0,
    "昌平社区": 240.0,
    "大兴社区": 200.0,
}


def get_community_geo(name: str) -> Dict[str, float]:
    """返回社区地理信息 dict，找不到则返回北京中心。"""
    lon, lat = COMMUNITY_LONGLAT.get(name, BEIJING_CENTER)
    return {
        "name": name,
        "lon": float(lon),
        "lat": float(lat),
        "base_height": float(BASE_HEIGHT_BY_COMMUNITY.get(name, 300.0)),
    }


def all_community_geo() -> Dict[str, Dict[str, float]]:
    """返回所有社区的地理信息 dict。"""
    return {name: get_community_geo(name) for name in COMMUNITY_LONGLAT.keys()}
