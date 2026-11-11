"""좌표 변환(UTM-K ↔ WGS84)과 행정동 경계 로딩.

pyproj가 없는 환경이라 EPSG:5179(Korea 2000 / Unified CS) 횡메르카토르를
Snyder 공식으로 직접 구현했다. DS-032(50m 셀)의 UTMK_X/UTMK_Y를 지도에 올릴 때 쓴다.
"""
import json
import math
from functools import lru_cache

import numpy as np
from shapely.geometry import shape

from config import DONG_GEOJSON, DONGS

_A = 6378137.0
_F = 1 / 298.257222101
_E2 = 2 * _F - _F * _F
_EP2 = _E2 / (1 - _E2)
_LAT0 = math.radians(38.0)
_LON0 = math.radians(127.5)
_K0 = 0.9996
_FE, _FN = 1_000_000.0, 2_000_000.0


def _meridian_arc(phi):
    e4, e6 = _E2 ** 2, _E2 ** 3
    return _A * ((1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
                 - (3 * _E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
                 + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
                 - (35 * e6 / 3072) * np.sin(6 * phi))


_M0 = _meridian_arc(_LAT0)


def wgs84_to_utmk(lon, lat):
    phi = np.radians(np.asarray(lat, dtype=float))
    lam = np.radians(np.asarray(lon, dtype=float))
    n = _A / np.sqrt(1 - _E2 * np.sin(phi) ** 2)
    t = np.tan(phi) ** 2
    c = _EP2 * np.cos(phi) ** 2
    a = (lam - _LON0) * np.cos(phi)
    m = _meridian_arc(phi)
    x = _FE + _K0 * n * (a + (1 - t + c) * a ** 3 / 6
                         + (5 - 18 * t + t ** 2 + 72 * c - 58 * _EP2) * a ** 5 / 120)
    y = _FN + _K0 * (m - _M0 + n * np.tan(phi) * (
        a ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
        + (61 - 58 * t + t ** 2 + 600 * c - 330 * _EP2) * a ** 6 / 720))
    return x, y


def utmk_to_wgs84(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    e4, e6 = _E2 ** 2, _E2 ** 3
    m = _M0 + (y - _FN) / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    c1 = _EP2 * np.cos(phi1) ** 2
    t1 = np.tan(phi1) ** 2
    n1 = _A / np.sqrt(1 - _E2 * np.sin(phi1) ** 2)
    r1 = _A * (1 - _E2) / (1 - _E2 * np.sin(phi1) ** 2) ** 1.5
    d = (x - _FE) / (n1 * _K0)
    phi = phi1 - (n1 * np.tan(phi1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _EP2 - 3 * c1 ** 2) * d ** 6 / 720)
    lam = _LON0 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6
                   + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _EP2 + 24 * t1 ** 2) * d ** 5 / 120
                   ) / np.cos(phi1)
    return np.degrees(lam), np.degrees(phi)


@lru_cache(maxsize=1)
def load_dong_features():
    """SGIS 경계(4안 재사용)에 행안부 코드를 붙여 반환: [(adm_cd, name, geom, feature)]."""
    with open(DONG_GEOJSON, encoding="utf-8") as fp:
        geojson = json.load(fp)
    sgis_to_mois = {sgis: (mois, name) for mois, sgis, name in DONGS}
    out = []
    for feature in geojson["features"]:
        sgis_cd = str(feature["properties"]["ADM_CD"])
        if sgis_cd not in sgis_to_mois:
            continue
        mois_cd, name = sgis_to_mois[sgis_cd]
        out.append((mois_cd, name, shape(feature["geometry"]), feature))
    return out


def dong_centroids():
    """{행안부코드: (lon, lat)} — 다각형 내부 대표점."""
    result = {}
    for code, _, geom, _ in load_dong_features():
        point = geom.representative_point()
        result[code] = (point.x, point.y)
    return result
