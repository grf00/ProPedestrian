# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
#from .cuhk03 import CUHK03
#from .dukemtmcreid import DukeMTMC
from .market1501 import Market1501
#from .msmt17 import MSMT17
from yolov7.fast_reid_master.fastreid.data.datasets.AirportALERT import AirportALERT
from yolov7.fast_reid_master.fastreid.data.datasets.iLIDS import iLIDS
from yolov7.fast_reid_master.fastreid.data.datasets.pku import PKU
from yolov7.fast_reid_master.fastreid.data.datasets.prai import PRAI
from yolov7.fast_reid_master.fastreid.data.datasets.prid import PRID
from yolov7.fast_reid_master.fastreid.data.datasets.grid import GRID
from yolov7.fast_reid_master.fastreid.data.datasets.saivt import SAIVT
from yolov7.fast_reid_master.fastreid.data.datasets.sensereid import SenseReID
from yolov7.fast_reid_master.fastreid.data.datasets.sysu_mm import SYSU_mm
from yolov7.fast_reid_master.fastreid.data.datasets.thermalworld import Thermalworld
from yolov7.fast_reid_master.fastreid.data.datasets.pes3d import PeS3D
from yolov7.fast_reid_master.fastreid.data.datasets.caviara import CAVIARa
from yolov7.fast_reid_master.fastreid.data.datasets.viper import VIPeR
from yolov7.fast_reid_master.fastreid.data.datasets.lpw import LPW
from yolov7.fast_reid_master.fastreid.data.datasets.shinpuhkan import Shinpuhkan
from yolov7.fast_reid_master.fastreid.data.datasets.wildtracker import WildTrackCrop
from yolov7.fast_reid_master.fastreid.data.datasets.cuhk_sysu import cuhkSYSU

# Vehicle re-id datasets
from .veri import VeRi
from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
