import math
import cv2 as cv
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from model import Color, Point, DeformType
from image_data import ImageData, SourceType
from processor import ImageProcessor


@dataclass
class DistanceAggregator:
    muar_point: Point
    template_point: Point
    distance: float


class ProcessedDataFields:
    TEMPLATE_IMAGE = "Template"
    MUAR_IMAGE = "Muar"
    TEMPLATE_POINTS = "TemplatePoints"
    MUAR_POINTS = "MuarPoints"
    ALL_POINTS_BY_ROW = "AllPointsByRow"
    MIN_DISTANCES = "MinDistances"
    PERSENTILES = "Persentiles"


BY_DEFORM_MSG = {DeformType.noneDeform: "No defects",
                 DeformType.inDeform: "Concavity",
                 DeformType.outDeform: "Bulge"}


class AnalizatorBaseException(Exception):
    """ Базовый класс ошибок анализатора """


class AnalizatorAttributeError(AnalizatorBaseException):
    """ Переданы неправильные входные данные """


class Analizator:
    """ Складывает обработанное изображение с растром и анализирует данные """

    def __init__(self, base_raster: ImageData, over_raster: ImageData, processed_image: ImageData):
        if (
                base_raster.source is not SourceType.RASTER
                or over_raster.source is not SourceType.RASTER
                or processed_image.source is not SourceType.PROCESSED
        ):
            raise AnalizatorAttributeError("[!] Переданы неправильные входные данные "
                                           f"{base_raster.source} {over_raster.source} {processed_image.source}")
        if processed_image.image.ndim > 2:
            _processed_image = ImageProcessor.threshold(
                processed_image.image, 127)
        else:
            _processed_image = processed_image.image
        _processed_image = ImageProcessor.resize(
            _processed_image, 1000, 1000, interpolation=cv.INTER_AREA)
        self._base_raster = base_raster
        self._over_raster = over_raster
        self._processed_image = ImageData(
            _processed_image, SourceType.PROCESSED)
        self.processed_data = {}
        self._process()

    def _row_border_coord_template(self):
        t_points = self.template_points
        y_points = [point.coy for point in t_points]
        y_points = sorted(list(set(y_points)))
        h_half = (y_points[1] - y_points[0]) / 2
        ranges = [(i, [y_co - h_half, y_co + h_half])
                  for i, y_co in enumerate(y_points, start=1)]
        return ranges

    def _set_points_into_near_row(self):
        muar_rows = []
        template_ranges = self._row_border_coord_template()
        m_points = self.muar_points
        for row in template_ranges:
            for point in m_points:
                if point.coy >= row[1][0] and not point.coy >= row[1][1]:
                    muar_rows.append((row[0], point))
        template_rows = []
        t_points = self.template_points
        for row in template_ranges:
            for point in t_points:
                if point.coy >= row[1][0] and not point.coy >= row[1][1]:
                    template_rows.append((row[0], point))
        return template_rows, muar_rows

    def _sort_points_by_rows(self):
        self._set_points_into_near_row()
        t_points_rows_sorted, m_points_rows_sorted = self._set_points_into_near_row()
        t_points_by_rows = defaultdict(list)
        m_points_by_rows = defaultdict(list)
        for t_point in t_points_rows_sorted:
            t_points_by_rows[t_point[0]].append(t_point[1])
        for m_point in m_points_rows_sorted:
            m_points_by_rows[m_point[0]].append(m_point[1])
        self.processed_data[ProcessedDataFields.ALL_POINTS_BY_ROW] = {
            "T": t_points_by_rows, "M": m_points_by_rows}

    def _row_distance_aggregate(self, template_row_points: List[Point], muar_row_points: List[Point]):
        muar_to_template_dist_aggregates = []
        for mrp in muar_row_points:
            min_dist = math.inf
            t_point = None
            for trp in template_row_points:
                dist = math.dist(mrp.to_tuple(), trp.to_tuple())
                if dist < min_dist:
                    min_dist = dist
                    t_point = trp
            if t_point:
                muar_to_template_dist_aggregates.append(
                    DistanceAggregator(mrp, t_point, min_dist))
        return muar_to_template_dist_aggregates

    def _point_distance_analysis(self):
        distance_aggregators = []
        template_points_by_row: dict = self.processed_data[ProcessedDataFields.ALL_POINTS_BY_ROW]["T"]
        muar_points_by_row: dict = self.processed_data[ProcessedDataFields.ALL_POINTS_BY_ROW]["M"]
        selected_rows_count = min(
            [max(list(template_points_by_row.keys())), max(list(muar_points_by_row.keys()))])
        for i in range(selected_rows_count):
            distance_aggregators.extend(self._row_distance_aggregate(
                template_points_by_row[i], muar_points_by_row[i]))

        self.processed_data[ProcessedDataFields.MIN_DISTANCES] = distance_aggregators

    def _calc_persentiles(self):
        distance_aggregators = self.processed_data[ProcessedDataFields.MIN_DISTANCES]
        distances = [dist_agg.distance for dist_agg in distance_aggregators]
        persent50 = np.percentile(distances, 50)
        persent90 = np.percentile(distances, 90)
        persent99 = np.percentile(distances, 99)
        self.processed_data[ProcessedDataFields.PERSENTILES] = (
            persent50, persent90, persent99)

    def _process(self):
        template = ImageProcessor.masking(
            self._base_raster.image, self._over_raster.image)
        muar = ImageProcessor.masking(
            self._processed_image.image, self._over_raster.image)
        self.processed_data[ProcessedDataFields.TEMPLATE_IMAGE] = template
        self.processed_data[ProcessedDataFields.MUAR_IMAGE] = muar
        t_points = ImageProcessor.hull_points(template).centers
        m_points = ImageProcessor.hull_points(muar).centers
        self.processed_data[ProcessedDataFields.TEMPLATE_POINTS] = t_points
        self.processed_data[ProcessedDataFields.MUAR_POINTS] = m_points
        self._sort_points_by_rows()
        self._point_distance_analysis()
        self._calc_persentiles()

    @property
    def template_image(self):
        return self.processed_data[ProcessedDataFields.TEMPLATE_IMAGE]

    @property
    def muar_image(self):
        return self.processed_data[ProcessedDataFields.MUAR_IMAGE]

    @property
    def template_points(self) -> List[Point]:
        return self.processed_data[ProcessedDataFields.TEMPLATE_POINTS]

    @property
    def muar_points(self) -> List[Point]:
        return self.processed_data[ProcessedDataFields.MUAR_POINTS]

    @property
    def distanses(self):
        return self.processed_data[ProcessedDataFields.MIN_DISTANCES]

    @property
    def persentiles(self):
        return self.processed_data[ProcessedDataFields.PERSENTILES]

    def has_deform(self):
        persentiles = self.processed_data[ProcessedDataFields.PERSENTILES]
        persentiles50 = persentiles[0]
        persentiles99 = persentiles[2]
        if persentiles50 > 4:
            return True
        if persentiles99 / persentiles50 > 2.1:
            return True
        return False

    def _poster_select_great_heights(self, poster: np.ndarray):
        color = Color.Yellow
        select_on = self.processed_data[ProcessedDataFields.PERSENTILES][1]
        for dist_aggregate in self.processed_data[ProcessedDataFields.MIN_DISTANCES]:
            dist_aggregate: DistanceAggregator
            if dist_aggregate.distance >= select_on:
                point1 = dist_aggregate.muar_point.to_tuple()
                point2 = dist_aggregate.template_point.to_tuple()
                cv.line(poster, point1, point2, color, 1)

    def poster(self, select_persentile90=True):
        poster_shape = self._processed_image.shape()
        poster_shape = (poster_shape[0], poster_shape[1], 3)
        poster = np.zeros(poster_shape, dtype='uint8')
        t_hull_group = self.template_points
        m_hull_group = self.muar_points
        t_color = Color.Green
        m_color = Color.Red
        for t_point in t_hull_group:
            cv.circle(poster, t_point.to_tuple(), 1, t_color, -1)
        for m_point in m_hull_group:
            cv.circle(poster, m_point.to_tuple(), 1, m_color, -1)
        if select_persentile90:
            self._poster_select_great_heights(poster)

        return poster
