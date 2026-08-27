from abc import ABC, abstractmethod
from datetime import datetime, UTC
from enum import StrEnum
from functools import cached_property
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from bc_caldb.constants import CURRENT_CALDB_VER, DEFAULT_MASK_SEED


MANDATORY_KEYWORDS = {
    "TELESCOP": ("BLACKCAT", "Telescope (mission) name"),
    "INSTRUME": ("BLACKCAT", "Instrument Name"),
    "DATE": (
        datetime.now(tz=UTC).strftime(f"%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "Creation Date",
    ),
    # Per-File
    # - CHECKSUM
    # - DATASUM
}
MANDATORY_TABLE_KEYWORDS = {
    "ORIGIN": ("PENNSTATE", "Source of FITS file"),
    "CREATOR": ("BC_CALDB", "Software that created FITS file"),
    # Per-File / Per-header
    # - EXTNAME
    # - CONTENT
    # - FILENAME
    # - VERSION
    # - CCLSxxxx
    # - CDTPxxxx
    # - CCNMxxxx
    # - CDESxxxx
    # - CVSDxxxx
    # - CVSTxxxx
}
# Situationally required
# - CBDnxxxx
# - CSYSNAME
# - TDIMnnn
# - HDUCLASS
# - HDUDOC
# - HDUCLASn
# - HDUVERSn
# - TIMESYS
# - MJDREFI
# - MJDREFF
# - CLOCKAPP


class CalDBVersions(StrEnum):
    DEFAULT = ""
    V20260614 = "20260614"


class GenerateCalDB(ABC):
    def __init__(self, caldb_version: Optional[str] = CURRENT_CALDB_VER):
        if not isinstance(caldb_version, str | None):
            raise TypeError(
                f"expected caldb_version to be str or None, got {type(caldb_version)}"
            )
        self._caldb_version = caldb_version if caldb_version is not None else ""
        if self._caldb_version not in CalDBVersions:
            raise ValueError(f"invalid caldb_version: {self._caldb_version}")
        self.generate_caldb_values()

    @cached_property
    def generation_keywords(self) -> dict[str, Any]:
        generation_keywords = {
            key.lower(): val[0]
            for tuple in self._commented_dicts
            for key, val in tuple[0].items()
        }

        return generation_keywords

    @abstractmethod
    def generate_caldb_values(self) -> None:
        self._commented_dicts: list[tuple[dict[str, tuple[Any, str]], str]]

    def generate_fits_file(self) -> None:
        # TODO: Implement this generation
        pass


class GenerateTeldef(GenerateCalDB):
    DET_IDS = [0, 1, 2, 3]
    DET_PITCH_M = 40e-6
    NOMINAL_GAP_M = 1788e-6
    NUM_SUBPIXELS = 3
    RAW_SIZE = 550

    def __init__(self, caldb_version: Optional[str] = CURRENT_CALDB_VER):
        super().__init__(caldb_version)

    @cached_property
    def _c(self) -> float:
        return self.DET_PITCH_M * (self.RAW_SIZE - 1 / 6) + self.NOMINAL_GAP_M / 2

    @cached_property
    def _det_offsets_dict(self) -> dict[str, dict[str, npt.NDArray[np.float32]]]:
        det_offsets_dict = {
            CalDBVersions.DEFAULT: {
                "x": np.array([0, 0, 0, 0], dtype=np.float32),
                "y": np.array([0, 0, 0, 0], dtype=np.float32),
            },
            CalDBVersions.V20260614: {
                "x": np.array([32.9e-6, 3.1e-6, -232.2e-6, 195.9e-6], dtype=np.float32),
                "y": np.array(
                    [189.0e-6, -167.7e-6, 96.2e-6, -117.6e-6], dtype=np.float32
                ),
            },
        }

        return det_offsets_dict

    @cached_property
    def _dx_dcols(self) -> npt.NDArray[np.float32]:
        return np.array(
            [
                -self.DET_PITCH_M / self.NUM_SUBPIXELS,
                self.DET_PITCH_M / self.NUM_SUBPIXELS,
                0,
                0,
            ],
            dtype=np.float32,
        )

    @cached_property
    def _dx_drows(self) -> npt.NDArray[np.float32]:
        return np.array(
            [
                0,
                0,
                self.DET_PITCH_M / self.NUM_SUBPIXELS,
                -self.DET_PITCH_M / self.NUM_SUBPIXELS,
            ],
            dtype=np.float32,
        )

    @cached_property
    def _dy_dcols(self) -> npt.NDArray[np.float32]:
        return np.array(
            [
                0,
                0,
                -self.DET_PITCH_M / self.NUM_SUBPIXELS,
                self.DET_PITCH_M / self.NUM_SUBPIXELS,
            ],
            dtype=np.float32,
        )

    @cached_property
    def _dy_drows(self) -> npt.NDArray[np.float32]:
        return np.array(
            [
                -self.DET_PITCH_M / self.NUM_SUBPIXELS,
                self.DET_PITCH_M / self.NUM_SUBPIXELS,
                0,
                0,
            ],
            dtype=np.float32,
        )

    @cached_property
    def _x0s(self) -> npt.NDArray[np.float32]:
        base_x0 = np.array([self._c, -self._c, -self._c, self._c], dtype=np.float32)
        return base_x0 + self._det_offsets_dict[self._caldb_version]["x"]

    @cached_property
    def _y0s(self) -> npt.NDArray[np.float32]:
        base_y0 = np.array([self._c, -self._c, self._c, -self._c], dtype=np.float32)
        return base_y0 + self._det_offsets_dict[self._caldb_version]["y"]

    def generate_caldb_values(self) -> None:
        self._commented_dicts = [
            (
                {
                    "CCLS0001": ("BCF", "Dataset is Basic Calibration File"),
                    "CCNM0001": ("TELDEF", "Type of calibration data"),
                    "CDTP0001": ("DATA", "Calibration file contains data"),
                    "CVSD0001": (
                        "2026-02-11",
                        "UTC date when calibration should first be used",
                    ),
                    "CVST0001": (
                        "00:00:00",
                        "UTC time when calibration should first be used",
                    ),
                    "CDES0001": ("TELESCOPE DEFINITION FILE", "Description"),
                },
                "",
            ),
            (
                {
                    "NCOORDS": (2, "Number of coordinates defined in this file"),
                    "COORD0": ("RAW", "1st coordinate system (DETID, RAWX, RAWY)"),
                    "COORD1": ("DET", "2nd coordinate system (DETX, DETY)"),
                    "COORD2": ("SAT", "3rd coordinate system (SATX, SATY, SATZ)"),
                    "COORD3": ("SKY", "4th coordinate system (SKYX, SKYY)"),
                },
                "",
            ),
            (
                {
                    "DET_IDS": (self.DET_IDS, "IDs of included detectors."),
                    "RAW_XSIZ": (
                        self.RAW_SIZE * self.NUM_SUBPIXELS,
                        "RAW address space x size (1/3 subpixels)",
                    ),
                    "RAWXPIX1": (
                        0.0,
                        "RAW address space x first subpixel number (1/3 subpixel)",
                    ),
                    "RAW_XSCL": (
                        self.DET_PITCH_M / self.NUM_SUBPIXELS,
                        "RAW X scale (m / subpixel)",
                    ),
                    "RAW_XCOL": ("RAWX", "Name of raw X column in event files"),
                    "RAW_YSIZ": (
                        self.RAW_SIZE * self.NUM_SUBPIXELS,
                        "RAW address space y size (1/3 subpixels)",
                    ),
                    "RAWYPIX1": (
                        0.0,
                        "RAW address space y first subpixel number (1/3 subpixel)",
                    ),
                    "RAW_YSCL": (
                        self.DET_PITCH_M / self.NUM_SUBPIXELS,
                        "RAW Y scale (m / subpixel)",
                    ),
                    "RAW_YCOL": ("RAWY", "Name of raw Y column in event files"),
                    "RAW_UNIT": ("1/3 subpixel", "physical unit of RAW coordinates"),
                },
                "",
            ),
            (
                {
                    "DETX_MIN": (np.min(self._x0s), "Minimum det x coordinate"),
                    "DETX_MAX": (np.max(self._x0s), "Maximum det x coordinate"),
                    "DET_XCOL": ("DETX", "Name of DET X column in event files"),
                    "DETY_MIN": (np.min(self._y0s), "Minimum det y coordinate"),
                    "DETY_MAX": (np.max(self._y0s), "Maximum det y coordinate"),
                    "DET_YCOL": ("DETY", "Name of DET Y column in event files"),
                    "DET_UNIT": ("m", "physical unit of DET coordinates"),
                },
                "",
            ),
            (
                {
                    "DET0_X0": (self._x0s[0],),
                    "DET0_Y0": (self._y0s[0],),
                    "DET0_DX_DCOL": (self._dx_dcols[0],),
                    "DET0_DY_DCOL": (self._dy_dcols[0],),
                    "DET0_DX_DROW": (self._dx_drows[0],),
                    "DET0_DY_DROW": (self._dy_drows[0],),
                    "DET1_X0": (self._x0s[1],),
                    "DET1_Y0": (self._y0s[1],),
                    "DET1_DX_DCOL": (self._dx_dcols[1],),
                    "DET1_DY_DCOL": (self._dy_dcols[1],),
                    "DET1_DX_DROW": (self._dx_drows[1],),
                    "DET1_DY_DROW": (self._dy_drows[1],),
                    "DET2_X0": (self._x0s[2],),
                    "DET2_Y0": (self._y0s[2],),
                    "DET2_DX_DCOL": (self._dx_dcols[2],),
                    "DET2_DY_DCOL": (self._dy_dcols[2],),
                    "DET2_DX_DROW": (self._dx_drows[2],),
                    "DET2_DY_DROW": (self._dy_drows[2],),
                    "DET3_X0": (self._x0s[3],),
                    "DET3_Y0": (self._y0s[3],),
                    "DET3_DX_DCOL": (self._dx_dcols[3],),
                    "DET3_DY_DCOL": (self._dy_dcols[3],),
                    "DET3_DX_DROW": (self._dx_drows[3],),
                    "DET3_DY_DROW": (self._dy_drows[3],),
                },
                "",
            ),
            (
                {
                    "SAT_UNIT": ("m", "physical unit of SAT coordinates"),
                },
                "",
            ),
            (
                {
                    "ALIGNM11": (0.0, "DET - > SAT coordinates alignment matrix Mij"),
                    "ALIGNM12": (0.0, ""),
                    "ALIGNM13": (-1.0, ""),
                    "ALIGNM21": (0.0, "[3x3 rotation matrix from focal plane to SAT]"),
                    "ALIGNM22": (-1.0, ""),
                    "ALIGNM23": (0.0, "SATX = M11*DETX + M12*DETY + M13*DETZ"),
                    "ALIGNM31": (-1.0, "SATY = M21*DETX + M22*DETY + M23*DETZ"),
                    "ALIGNM32": (0.0, "SATZ = M31*DETX + M32*DETY + M33*DETZ"),
                    "ALIGNM33": (0.0, ""),
                    "ROLLSIGN": (-1, "BlackCAT Roll convention"),
                },
                "",
            ),
            # TODO: Implement SKY coordinates and SAT -> SKY transforms
            ({"FOCALLEN": (0.1540, "Telescope focal length (m)")}, ""),
            (
                {
                    "OPTAXISX": (0.0, "Optical axis x in DET coordinates (m)"),
                    "OPTAXISY": (0.0, "Optical axis y in DET coordinates (m)"),
                },
                "",
            ),
        ]


class GenerateCodedMask(GenerateCalDB):
    MASK_CELL_COUNT = [249, 555]

    def __init__(self, caldb_version: Optional[str] = CURRENT_CALDB_VER):
        super().__init__(caldb_version)

    @cached_property
    def _det_cent_sat_xy(self) -> tuple[float, float]:
        generated_teldef = GenerateTeldef(self._caldb_version)
        detx_center = (
            generated_teldef.generation_keywords["detx_max"]
            - generated_teldef.generation_keywords["detx_min"]
        ) / 2
        dety_center = (
            generated_teldef.generation_keywords["dety_max"]
            - generated_teldef.generation_keywords["dety_min"]
        ) / 2
        return detx_center, dety_center

    @cached_property
    def _frame_pattern(self) -> npt.NDArray[np.bool_]:
        pattern = np.zeros(shape=self.MASK_CELL_COUNT, dtype=bool)

        # Main ribs
        for xlow, ylow, xblocksize, yblocksize in self._ribs:
            pattern[ylow : ylow + yblocksize, xlow : xlow + xblocksize] = True

        # Manual fillets
        for xrib_l in [-4, self.MASK_CELL_COUNT[0] // 2, self.MASK_CELL_COUNT[0] + 3]:
            for yrib_l in [-4, self.MASK_CELL_COUNT[1] + 3] + list(
                self._yribx.astype(int)
            ):
                pattern[
                    max(xrib_l - 4, 0) : xrib_l + 5, max(yrib_l - 7, 0) : yrib_l + 8
                ] = True
                pattern[
                    max(xrib_l - 7, 0) : xrib_l + 8, max(yrib_l - 4, 0) : yrib_l + 5
                ] = True
                pattern[
                    max(xrib_l - 5, 0) : xrib_l + 6, max(yrib_l - 5, 0) : yrib_l + 6
                ] = True

        # Screwhole gussets
        for yrib_l in self._yribx.astype(int):
            for y, ysign in [(0, 1), (self.MASK_CELL_COUNT[0] - 1, -1)]:
                for dy, w in enumerate([19, 15, 13, 11, 9]):
                    pattern[y + dy * ysign, yrib_l - w // 2 : yrib_l + w // 2 + 1] = (
                        True
                    )

        return pattern

    @cached_property
    def generation_keywords(self) -> dict[str, Any]:
        generation_keywords = super().generation_keywords
        generation_keywords["mask_pattern"] = self._mask_pattern
        generation_keywords["frame_pattern"] = self._frame_pattern

        return generation_keywords

    @cached_property
    def _mask_pattern(self) -> npt.NDArray[np.bool_]:
        ny, nx = self.MASK_CELL_COUNT

        pattern = (
            np.array(self.shift_reg_seq(nx * ny, DEFAULT_MASK_SEED), dtype=bool)
            .reshape(self.MASK_CELL_COUNT[::-1])
            .T
        )

        return pattern[::-1, :] & ~self._frame_pattern

    @cached_property
    def _ribs(self) -> npt.NDArray[np.uint16]:
        ribhw = 3.5
        xriby = self.MASK_CELL_COUNT[0] / 2
        xcellcount = self.MASK_CELL_COUNT[1]
        yribx = self._yribx
        ycellcount = self.MASK_CELL_COUNT[0]

        ribs = np.array(
            [
                (yribx[0] - ribhw, 0, 2 * ribhw, ycellcount),
                (yribx[1] - ribhw, 0, 2 * ribhw, ycellcount),
                (yribx[2] - ribhw, 0, 2 * ribhw, ycellcount),
                (0, xriby - ribhw, xcellcount, 2 * ribhw),
            ],
            dtype=np.uint16,
        )

        return ribs

    @cached_property
    def _yribx(self) -> npt.NDArray[np.float32]:
        return np.array([138.5, 277.5, 416.5], dtype=np.float32)

    def generate_caldb_values(self) -> None:
        self._commented_dicts = [
            (
                {
                    "CCLS0001": ("BCF", "Dataset is Basic Calibration File"),
                    "CCNM0001": ("CODED_MASK", "Type of calibration data"),
                    "CDTP0001": ("DATA", "Calibration file contains data"),
                    "CVSD0001": (
                        "2026-02-11",
                        "UTC date when calibration should first be used",
                    ),
                    "CVST0001": (
                        "00:00:00",
                        "UTC time when calibration should first be used",
                    ),
                    "CDES0001": (
                        "BlackCAT Coded mask (aperture) pattern",
                        "Description",
                    ),
                },
                "",
            ),
            (
                {
                    "CTYPE1": ("SATX", "Title of this axis"),
                    "CRPIX1": (-0.5, "Reference is lower left corner of mask"),
                    "CRVAL1": (
                        -self.MASK_CELL_COUNT[1] * 320e-6 / 2,
                        "Value of SATX at reference point",
                    ),
                    "CRUNIT1": ("m", "Units of SATX"),
                    "CDELT1": (320e-6, "Spacing of cells in m"),
                    "CTYPE2": ("SATY", "Title of this axis"),
                    "CRPIX2": (-0.5, "Reference is lower left corner of mask)"),
                    "CRVAL2": (
                        -self.MASK_CELL_COUNT[0] * 320e-6 / 2,
                        "Value of SATY at reference point",
                    ),
                    "CRUNIT2": ("m", "Units of SATY"),
                    "CDELT2": (320e-6, "Spacing of cells in m"),
                },
                "",
            ),
            (
                {
                    "MASKSATX": (-0.1540, "[m] Center of mask cell plane in SATX"),
                    "MASKSATY": (0.0, "[m] Center of mask cell plane in SATY"),
                    "MASKSATZ": (0.0, "[m] Top of mask cell plane in SATZ"),
                    "MASKOFFX": (0.0, "[m] Offset of mask in SATX"),
                    "MASKOFFY": (0.0, "[m] Offset of mask in SATY"),
                    "MASKOFFZ": (0.0, "[m] Offset of mask in SATZ"),
                    "MASKPSIX": (0.0, "[deg] Mask Euler rotation about X-axis"),
                    "MASKPSIY": (0.0, "[deg] Mask Euler rotation about Y-axis"),
                    "MASKPSIZ": (0.0, "[deg] Mask Euler rotation about Z-axis"),
                },
                "",
            ),
            (
                {
                    "MASKCELX": (295e-6, "[m] Size of mask cell in SATX"),
                    "MASKCELY": (295e-6, "[m] Size of mask cell in SATY"),
                    "MASKCELZ": (21e-6, "[m] Size of mask cell in SATZ"),
                },
                "",
            ),
            (
                {
                    "DETSATX": (0.0, "[m] Top of detector plane in SATX"),
                    "DETSATY": (
                        -self._det_cent_sat_xy[1],
                        "[m] Center of detector plane in SATY",
                    ),
                    "DETSATZ": (
                        -self._det_cent_sat_xy[0],
                        "[m] Center of detector plane in SATZ",
                    ),
                    "DETOFFX": (0.0, "[m] Offset of detector plane in SATX"),
                    "DETOFFY": (0.0, "[m] Offset of detector plane in SATY"),
                    "DETOFFZ": (0.0, "[m] Offset of detector plane in SATZ"),
                },
                "",
            ),
            (
                {
                    "DETPIXX": (40e-6, "[m] Size of detector pitch pixel in SATX"),
                    "DETPIXY": (40e-6, "[m] Size of detector pitch pixel in SATX"),
                    "DETPIXZ": (100e-6, "[m] Size of detector pitch pixel in SATX"),
                    "DETSIZEX": (40e-6, "[m] Size of detector pixel in SATX"),
                    "DETSIZEY": (40e-6, "[m] Size of detector pixel in SATX"),
                    "DETSIZEZ": (100e-6, "[m] Size of detector pixel in SATX"),
                },
                "",
            ),
        ]

    @staticmethod
    def shift_reg_seq(seql: int, seed: int) -> npt.NDArray[np.bool_]:
        nbits = int(np.ceil(np.log2(seql + 1)))

        taplist = [
            -1,
            -1,
            -1,
            2,
            3,
            3,
            5,
            6,
            -1,
            5,
            7,
            9,
            -1,
            -1,
            -1,
            14,
            -1,
            14,
            11,
            -1,
            17,
            19,
            21,
            18,
            -1,
            22,
            -1,
            -1,
            25,
            27,
            -1,
            28,
        ]
        if taplist[nbits] == -1:
            raise RuntimeError(
                f"No single-tap maximal LFSR with {nbits} bits: specify multiple taps"
            )
        taps = [nbits, taplist[nbits]]

        seq = np.zeros(seql, dtype=bool)
        for idx, seedbit in enumerate(reversed(f"{seed:b}")):
            if idx >= nbits:
                break
            seq[idx] = seedbit == "1"
        for idx in range(nbits, seql):
            for tap in taps:
                seq[idx] ^= seq[idx - tap]

        return seq
