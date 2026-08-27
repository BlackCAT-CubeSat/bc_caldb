from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from bc_caldb.caldb_generators import (
    GenerateCalDB,
    GenerateCodedMask,
    GenerateTeldef,
)
from bc_caldb.constants import CURRENT_CALDB_VER


@dataclass(frozen=True)
class CalDB(ABC):
    @classmethod
    def from_caldb_file(cls, caldb_file: PathLike | str):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_caldb_version(
        cls, generator: GenerateCalDB, caldb_version: str = CURRENT_CALDB_VER
    ):
        caldb_generator = generator(caldb_version)
        return cls(**caldb_generator.generation_keywords)


@dataclass(frozen=True)
class Teldef(CalDB):
    ccls0001: str
    ccnm0001: str
    cdtp0001: str
    cvsd0001: str
    cvst0001: str
    cdes0001: str

    ncoords: int
    coord0: str
    coord1: str
    coord2: str
    coord3: str

    det_ids: list[int]
    raw_xsiz: float
    rawxpix1: float
    raw_xscl: float
    raw_xcol: str
    raw_ysiz: float
    rawypix1: float
    raw_yscl: float
    raw_ycol: str
    raw_unit: str

    detx_min: float
    detx_max: float
    det_xcol: str
    dety_min: float
    dety_max: float
    det_ycol: str
    det_unit: str

    det0_x0: float
    det0_y0: float
    det0_dx_dcol: float
    det0_dy_dcol: float
    det0_dx_drow: float
    det0_dy_drow: float
    det1_x0: float
    det1_y0: float
    det1_dx_dcol: float
    det1_dy_dcol: float
    det1_dx_drow: float
    det1_dy_drow: float
    det2_x0: float
    det2_y0: float
    det2_dx_dcol: float
    det2_dy_dcol: float
    det2_dx_drow: float
    det2_dy_drow: float
    det3_x0: float
    det3_y0: float
    det3_dx_dcol: float
    det3_dy_dcol: float
    det3_dx_drow: float
    det3_dy_drow: float

    sat_unit: str

    alignm11: float
    alignm12: float
    alignm13: float
    alignm21: float
    alignm22: float
    alignm23: float
    alignm31: float
    alignm32: float
    alignm33: float
    rollsign: int

    focallen: float

    optaxisx: int
    optaxisy: int

    def detxyz_to_satxyz(
        self,
        detxs: npt.NDArray[np.floating[Any]],
        detys: npt.NDArray[np.floating[Any]],
        detzs: Optional[npt.NDArray[np.floating[Any]]] = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if detzs is None:
            detzs = np.zeros(detxs.shape, dtype=np.float64)

        satx = (
            detxs * self.alignm11 + detys * self.alignm12 + detzs * self.alignm13
        ).astype(np.float64)
        saty = (
            detxs * self.alignm21 + detys * self.alignm22 + detzs * self.alignm23
        ).astype(np.float64)
        satz = (
            detxs * self.alignm31 + detys * self.alignm32 + detzs * self.alignm33
        ).astype(np.float64)

        return satx, saty, satz

    def rawxy_to_detxy(
        self,
        rawxs: npt.NDArray[np.floating[Any] | np.integer[Any]],
        rawys: npt.NDArray[np.floating[Any] | np.integer[Any]],
        detids: npt.NDArray[np.floating[Any] | np.integer[Any]],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        detxs = np.zeros(rawxs.shape, dtype=np.float64)
        detys = np.zeros(rawys.shape, dtype=np.float64)

        for detid in self.det_ids:
            detid_mask = detids == detid
            detxs[detid_mask] = (
                getattr(self, f"det{detid}_x0")
                + rawxs[detid_mask] * getattr(self, f"det{detid}_dx_dcol")
                + rawys[detid_mask] * getattr(self, f"det{detid}_dx_drow")
            )
            detys[detid_mask] = (
                getattr(self, f"det{detid}_y0")
                + rawxs[detid_mask] * getattr(self, f"det{detid}_dy_dcol")
                + rawys[detid_mask] * getattr(self, f"det{detid}_dy_drow")
            )

        return detxs, detys

    @classmethod
    def from_caldb_version(cls, caldb_version: str = CURRENT_CALDB_VER):
        return super().from_caldb_version(GenerateTeldef, caldb_version)


@dataclass(frozen=True)
class CodedMask(CalDB):
    ccls0001: str
    ccnm0001: str
    cdtp0001: str
    cvsd0001: str
    cvst0001: str
    cdes0001: str

    ctype1: str
    crpix1: float
    crval1: float
    crunit1: str
    cdelt1: float
    ctype2: str
    crpix2: float
    crval2: float
    crunit2: str
    cdelt2: float

    masksatx: float
    masksaty: float
    masksatz: float
    maskoffx: float
    maskoffy: float
    maskoffz: float
    maskpsix: float
    maskpsiy: float
    maskpsiz: float

    maskcelx: float
    maskcely: float
    maskcelz: float

    detsatx: float
    detsaty: float
    detsatz: float
    detoffx: float
    detoffy: float
    detoffz: float

    detpixx: float
    detpixy: float
    detpixz: float
    detsizex: float
    detsizey: float
    detsizez: float

    mask_pattern: npt.NDArray[np.bool_]
    frame_pattern: npt.NDArray[np.bool_]

    @classmethod
    def from_caldb_version(cls, caldb_version: str = CURRENT_CALDB_VER):
        return super().from_caldb_version(GenerateCodedMask, caldb_version)
