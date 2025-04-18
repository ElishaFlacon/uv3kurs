import cv2 as cv
import numpy as np
import math
from model import Point, Section
from settings import WindowSettings, RasterSettings
from paths import save_data


def _get_line_shift(distance, angle):
    sin = math.sin(math.radians(angle + 90))
    cos = math.cos(math.radians(angle + 90))
    return int(distance * cos), int(distance * sin)


def _get_lines_amount(center: Point, distance):
    return 2 * int(Point.distance(Point(0, 0), center) / distance)


class RasterFactory:
    def __init__(self, win_settings: WindowSettings, raster_settings: RasterSettings, use_save=True):
        self.use_save = use_save
        self.center = win_settings.center
        self.settings = raster_settings
        self._raster = np.zeros(
            (self.center.coy * 2, self.center.cox * 2), dtype='uint8')
        self.amount = _get_lines_amount(
            self.center, raster_settings.distance + raster_settings.thickness) or 1

    @property
    def raster(self):
        return self._raster

    @raster.setter
    def raster(self, value):
        self._raster = value

    def process(self):
        angle = self.settings.angle
        distance = self.settings.distance
        thickness = self.settings.thickness
        offset = self.settings.offset
        color = self.settings.color
        length = math.ceil(2 * Point.distance(Point(0, 0), self.center))

        xshift, yshift = _get_line_shift(distance, angle)
        xoffset, yoffset = _get_line_shift(offset, angle)
        high_point = Point(self.center.cox + self.amount *
                           xshift, self.center.coy + self.amount * yshift)
        normal = Section(self.center, high_point)
        perp = Section.perp(normal, length)

        for i in range(self.amount, -self.amount, -1):
            pta = (perp.pta.cox + xoffset, perp.pta.coy + yoffset)
            ptb = (perp.ptb.cox + xoffset, perp.ptb.coy + yoffset)
            cv.line(self._raster, pta, ptb, color, thickness)
            perp.shift(xshift, yshift)

        return self._raster

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_save:
            save_data(self._raster, self.settings.stringify())
