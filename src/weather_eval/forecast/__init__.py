from .base import ForecastProvider
from .open_meteo import OpenMeteoProvider
from .caiyun import CaiyunProvider
from .qweather import QWeatherProvider
from .tianji import TianjiProvider
from .fuxi import FuxiC88Provider
from .fuxi_data import FuxiDetProvider
from .fengwu import FengWuProvider
from .geovis import GevisProvider
from .accuweather import AccuWeatherProvider
from .msn import MsnProvider
from .ew4all import Ew4allProvider

__all__ = [
    "ForecastProvider", "OpenMeteoProvider", "CaiyunProvider", "QWeatherProvider",
    "TianjiProvider", "FuxiC88Provider", "FuxiDetProvider", "FengWuProvider",
    "GevisProvider", "AccuWeatherProvider", "MsnProvider", "Ew4allProvider",
]
