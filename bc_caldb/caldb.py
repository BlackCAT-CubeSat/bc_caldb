from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Optional

from astropy.io import fits
import numpy as np
import numpy.typing as npt

from bc_caldb.constants import CURRENT_CALDB_VER, DEFAULT_MASK_SEED


@dataclass
class Teldef:
    # TODO: Add additional elements
    #   - DET -> SAT transform
    #   - SKY
    #   - SKY -> SAT translation
    #   - Generation of caldb fits file from class

    CCLS0001 = "BCF"
    CCNM0001 = "TELDEF"
    CDTP0001 = "DATA"
    CVSD0001 = "2026-02-11"
    CVST0001 = "00:00:00"
    CDES0001 = "TELESCOPE DEFINITION FILE"

    NCOORDS = 3
    COORD0 = "RAW"
    COORD1 = "DET"
    COORD2 = "SKY"

    RAW_XSIZ = 1650
    RAWXPIX1 = 0.0
    RAW_XSCL = 40e-6 / 3
    RAW_XCOL = "RAWX"
    RAW_YSIZ = 1650
    RAWYPIX1 = 0.0
    RAW_YSCL = 40e-6 / 3
    RAW_YCOL = "RAWY"
    RAW_UNIT = "1/3 subpixel"

    NUM_DETS = 4
    DET_XCOL = "DETX"
    DET_YCOL = "DETY"
    DET_UNIT = "m"
    detx_min: float 
    detx_max: float
    dety_min: float
    dety_max: float

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

    FOCALLEN = (0.1540, "Telescope focal length (m)")

    # The following aren't used externally
    _CONVERT_DTYPE = [
        ("x0", np.float32),
        ("y0", np.float32),
        ("dx_dcol", np.float32),
        ("dy_dcol", np.float32),
        ("dx_drow", np.float32),
        ("dy_drow", np.float32),
    ]

    @property
    def det_ids(self) -> np.ndarray[np.uint8]:
        return np.arange(self.NUM_DETS, dtype=np.uint8)

    def det_envelope(self) -> npt.NDArray[np.float64]:
        detx, dety = self.rawxy_to_detxy(
            np.array([0, 0, 0, 0], dtype=np.uint8),
            np.array([0, 0, 0, 0], dtype=np.uint8),
            self.det_ids,
        )

        det_envelope = np.array([
            [np.min(detx), np.min(dety)],
            [np.max(detx), np.max(dety)],
        ], dtype=np.float64)

        return det_envelope

    def rawxy_to_detxy(
        self,
        rawx_arr: npt.NDArray[np.integer[Any]],
        rawy_arr: npt.NDArray[np.integer[Any]],
        det_id_arr: npt.NDArray[np.integer[Any]],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        convert_array = np.array([
            (self.det0_x0, self.det0_y0, self.det0_dx_dcol, self.det0_dy_dcol, self.det0_dx_drow, self.det0_dy_drow),
            (self.det1_x0, self.det1_y0, self.det1_dx_dcol, self.det1_dy_dcol, self.det1_dx_drow, self.det1_dy_drow),
            (self.det2_x0, self.det2_y0, self.det2_dx_dcol, self.det2_dy_dcol, self.det2_dx_drow, self.det2_dy_drow),
            (self.det3_x0, self.det3_y0, self.det3_dx_dcol, self.det3_dy_dcol, self.det3_dx_drow, self.det3_dy_drow),
        ], dtype=self._CONVERT_DTYPE)

        detx_arr = np.zeros(len(rawx_arr), dtype=np.float64)
        dety_arr = np.zeros(len(rawx_arr), dtype=np.float64)

        for det_id in self.det_ids:
            det_id_mask = det_id_arr == det_id
            detx_arr[det_id_mask] = (
                convert_array[det_id]["x0"]
                + rawx_arr[det_id_mask]*convert_array[det_id]["dx_dcol"]
                + rawy_arr[det_id_mask]*convert_array[det_id]["dx_drow"]
            )
            dety_arr[det_id_mask] = (
                convert_array[det_id]["y0"]
                + rawx_arr[det_id_mask]*convert_array[det_id]["dy_dcol"]
                + rawy_arr[det_id_mask]*convert_array[det_id]["dy_drow"]
            )

        return detx_arr, dety_arr

    @classmethod
    def calculate_raw_det_conversion_values(
        cls, caldb_ver: Optional[str] = None
    ) -> npt.NDArray[np.float32]:
        nominal_gap = 1788e-6
        p_full = cls.RAW_XSCL * 3
        p = cls.RAW_XSCL
        c = ((cls.RAW_XSIZ / 3) - (1 / 6)) * p_full + nominal_gap / 2

        raw_det_conversion_arr = np.array(
            [
                (c, c, -p, 0, 0, -p),
                (-c, -c, p, 0, 0, p),
                (-c, c, 0, -p, p, 0),
                (c, -c, 0, p, -p, 0),
            ],
            dtype=cls._CONVERT_DTYPE,
        )

        match caldb_ver:
            case None:
                pass
            case "20260614":
                raw_det_conversion_arr["x0"] += np.array(
                    [32.9e-6, 3.1e-6, -232.2e-6, 195.9e-6]
                )
                raw_det_conversion_arr["y0"] += np.array(
                    [189.0e-6, -167.7e-6, 96.2e-6, -117.6e-6]
                )
            case str():
                raise ValueError(f"{caldb_ver} is not a valid CalDB version.")
            case _:
                raise TypeError(f"caldb_ver must be str, not {type(caldb_ver)}")

        return raw_det_conversion_arr

    @classmethod
    def from_caldb_version(cls, caldb_ver: Optional[str] = CURRENT_CALDB_VER):
        raw_det_conversion_arr = cls.calculate_raw_det_conversion_values(caldb_ver)

        teldef_obj = cls(
            0, 0, 0, 0,
            *raw_det_conversion_arr[0],
            *raw_det_conversion_arr[1],
            *raw_det_conversion_arr[2],
            *raw_det_conversion_arr[3],
        )

        det_mins, det_maxs = teldef_obj.det_envelope()
        teldef_obj.detx_min = det_mins[0]
        teldef_obj.dety_min = det_mins[1]
        teldef_obj.detx_max = det_maxs[0]
        teldef_obj.dety_max = det_maxs[1]

        return teldef_obj

@dataclass
class CodedMask:
    # TODO: Fill in coded mask details
    #   - Do we want separate types of masks?
    #   - Fill in remaining header keywords

    CCLS0001 = "BCF"
    CCNM0001 = "CODED_MASK"
    CDTP0001 = "DATA"
    CVSD0001 = "2026-02-11"
    CVST0001 = "00:00:00"
    CDES0001 = "BlackCAT Coded mask (aperture) pattern"

    # CTYPE1 = 
    # CRPIX1 = 
    # CRVAL1 = 
    CUNIT1 = "m"
    CDELT1 = 320e-6
    # CTYPE2 = 
    # CRPIX2 = 
    # CRVAL2 = 
    CUNIT2 = "m"
    CDELT2 = 320e-6

    # MASKSATX = 
    # MASKSATY = 
    MASKSATZ = 0.1540
    # MASKOFFX = 
    # MASKOFFY = 
    # MASKOFFZ = 

    # MASKPSI0 = 
    # MASKPSI1 = 
    # MASKPSI2 = =

    MASKCELX = 295e-6
    MASKCELY = 295e-6
    # MASKCELZ = 

    # DETSATX = 
    # DETSATY = 
    DETSATZ = 0
    # DETOFFX = 
    # DETOFFY = 
    # DETOFFZ = 

    DETCELX = 40e-6 / 3
    DETCELY = 40e-6 / 3
    # DETCELZ = 
    DETSIZEX = 40e-6 / 3
    DETSIZEY = 40e-6 / 3
    # DETSIZEZ = 

    mask_array: Optional[npt.NDArray | PathLike | str] = None

    # The following aren't used externally
    _MASK_CELL_COUNT = np.array([555, 249])
    _XRIBX = np.array([138.5, 277.5, 416.5])

    def frame_pattern(self) -> npt.NDArray[np.bool]:
        pattern = np.zeros(shape=self._MASK_CELL_COUNT, dtype=bool)

        for xlow, ylow, xblocksize, yblocksize, nsteps, xstep, ystep in self.ribs():
            for _ in range(nsteps):
                pattern[xlow:xlow+xblocksize, ylow:ylow+yblocksize] = True
                xlow += xstep
                ylow += ystep

        ymesh, xmesh = np.meshgrid(
            np.arange(self._MASK_CELL_COUNT[1]), np.arange(self._MASK_CELL_COUNT[0])
        )

        # Manually applying the fillets.
        # _l means centerline rounded down (i.e., lower)
        for yrib_l in [-4, self._MASK_CELL_COUNT[1] // 2, self._MASK_CELL_COUNT[1] + 3]:
            for xrib_l in [-4, self._MASK_CELL_COUNT[0] + 3] + list(self._XRIBX.astype(int)):
                pattern[max(xrib_l - 7, 0):xrib_l+5, max(yrib_l - 7, 0):yrib_l+5] = True
                pattern[max(xrib_l - 4, 0):xrib_l+8, max(yrib_l - 4, 0):yrib_l+8] = True
                pattern[max(xrib_l - 5, 0):xrib_l+6, max(yrib_l - 5, 0):yrib_l+6] = True

        screw_gus_widths = [19, 15, 13, 11, 9]
        for xrib_l in self._XRIBX.astype(int):
            for y, y_sign in [(0, 1), (self._MASK_CELL_COUNT[1] - 1, -1)]:
                for dy, w in enumerate(screw_gus_widths):
                    pattern[xrib_l - w//2:xrib_l + w//2 + 1, y + dy*y_sign] = True

        return pattern

    def mask_pattern(self) -> npt.NDArray[np.bool]:
        nx, ny = self._MASK_CELL_COUNT

        if self.mask_array is not None:
            if isinstance(self.mask_array, np.ndarray):
                maskpat_f = self.mask_array
            elif isinstance(self.mask_array, PathLike | str):
                maskpat_f = fits.getdata(Path(self.mask_array))
            else:
                raise TypeError(
                    "Mask array must be a numpy array or path to a fits "
                    f"file, not {type(self.mask_array)}."
                )

            pattern = (
                np.array(maskpat_f, dtype=bool)
                .ravel()[-nx*ny :]
                .reshape(self._MASK_CELL_COUNT)
                .copy()
            )   # Copy to release open mask file
        else:
            pattern = np.array(
                self.shift_register_sequence(seqlength=nx*ny, seed=DEFAULT_MASK_SEED),
                dtype=bool,
            ).reshape(self._MASK_CELL_COUNT)

            frame_pattern = self.frame_pattern()

            pattern = pattern & ~frame_pattern

        return pattern

    def ribs(self) -> npt.NDArray[np.integer[Any]]:
        yriby = self._MASK_CELL_COUNT[1] / 2
        ribhw = 3.5

        ribs = np.array(
            [
                (self._XRIBX[0] - ribhw, 0, 2 * ribhw, self._MASK_CELL_COUNT[1], 1, 0, 0),
                (self._XRIBX[1] - ribhw, 0, 2 * ribhw, self._MASK_CELL_COUNT[1], 1, 0, 0),
                (self._XRIBX[2] - ribhw, 0, 2 * ribhw, self._MASK_CELL_COUNT[1], 1, 0, 0),
                (0, yriby - ribhw, self._MASK_CELL_COUNT[0], 2 * ribhw, 1, 0, 0),
            ], dtype=int,
        )

        return ribs

    @staticmethod
    def shift_register_sequence(seqlength: int, seed: int) -> npt.NDArray[np.bool]:
        nbits = int(np.ceil(np.log2(seqlength + 1)))

        # From Horowitz and Hill pg. 657
        taplist = [
            -1, -1, -1, 2, 3, 3, 5, 6, -1, 5, 7, 9, -1, -1, -1, 14, -1, 14, 11, -1, 17, 19, 21, 18, -1, 22, -1, -1, 25, 27, -1, 28
        ]
        if taplist[nbits] == -1:
            raise RuntimeError(f"No single-tap maximal LFSR with {nbits} bits: specify multiple taps")
        taps = [nbits, taplist[nbits]]

        sequence = np.zeros(seqlength, dtype=bool)

        for idx, seedbit in enumerate(reversed(f"{seed:b}")):
            if idx >= nbits:
                break

            sequence[idx] = seedbit == "1"

        for idx in range(nbits, seqlength):
            for tap in taps:
                sequence[idx] ^= sequence[idx - tap]   # xor starting with False

        return sequence


@dataclass
class AncillaryResponse:
    # TODO: Fill in ARF details
    pass


@dataclass
class Response:
    # TODO: Fill in response details
    pass

@dataclass
class BadPix:
    # TODO: Fill in bad pixel mask details
    pass

@dataclass
class GainCorr:
    # TODO: Fill in gain correction details
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mask_obj = CodedMask()
    plt.imshow(mask_obj.mask_pattern(), origin="lower")
    plt.show()
