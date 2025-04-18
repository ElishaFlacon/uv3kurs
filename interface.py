import math
from numpy import ndarray
from dataclasses import dataclass
from typing import Optional, Any

import api
from settings import WindowSettings
from analysis import Analizator, BY_DEFORM_MSG
import dearpygui.dearpygui as dpg
import dearpygui.demo as demo


class Tag:
    WIN_STAT = "WIN_STAT"
    WIN_VIEW = "WIN_VIEW"
    WIN_INPUT = "WIN_INPUT"
    WIN_CONTROL = "WIN_CONTROL"
    WIN_MAIN_VIEW = "WIN_MAIN_VIEW"

    FILE_DIALOG_LOAD_BASE = "FILE_INPUT_LOAD_BASE"
    FILE_DIALOG_LOAD_OVER = "FILE_INPUT_LOAD_OVER"
    FILE_DIALOG_LOAD_RAW = "FILE_INPUT_LOAD_RAW"
    FILE_DIALOG_LOAD_PROCESS = "FILE_INPUT_LOAD_PROCESS"

    BTN_DATA_SHOW_CAMERA = "BTN_DATA_SHOW_CAMERA"
    BTN_INPUT_LOAD_BASE = "BTN_INPUT_LOAD_BASE"
    BTN_INPUT_LOAD_OVER = "BTN_INPUT_LOAD_OVER"
    BTN_INPUT_LOAD_RAW = "BTN_INPUT_LOAD_RAW"
    BTN_INPUT_LOAD_PROCESS = "BTN_INPUT_LOAD_PROCESS"
    BTN_INPUT_SHOW_RASTER = "BTN_INPUT_SHOW_RASTER"
    BTN_INPUT_RASTER_FACTORY = "BTN_INPUT_RASTER_FACTORY"
    BTN_CONTROL_ANALYSIS = "BTN_CONTROL_ANALYSIS"
    BTN_CONTROL_SHOW_ANALYSIS = "BTN_CONTROL_SHOW_ANALYSIS"
    BTN_CONTROL_MAKE_RAW = "BTN_CONTROL_MAKE_RAW"
    BTN_MAKE_RAW_PROCESS = "BTN_MAKE_RAW_PROCESS"

    GROUP_INPUT_PARENT = "GROUP_INPUT_PARENT"
    GROUP_INPUT_BTN_COLLECT = "GROUP_INPUT_BTN_COLLECT"
    GROUP_INPUT_INPUTS_COLLECT = "GROUP_INPUT_INPUTS_COLLECT"
    GROUP_CONTROL_BTN_COLLECT = "GROUP_CONTROL_BUTTON_COLLECTOR"
    GROUP_CONTROL_PROCESS_COLLECT = "GROUP_CONTROL_PROCESS_COLLECT"
    GROUP_DATA_TEXT_COLLECTOR = "GROUP_DATA_TEXT_COLLECTOR"

    TEXTURE_BASE = "TEXTURE_BASE"
    TEXTURE_OVER = "TEXTURE_OVER"
    TEXTURE_RAW = "TEXTURE_RAW"
    TEXTURE_PROCESS = "TEXTURE_PROCESS"
    TEXTURE_REG = "TEXTURE_REG"
    TEXTURE_CAMERA = "TEXTURE_CAMERA"
    TEXTURE_ANALIZATOR_POSTER = "TEXTURE_ANALIZATOR_POSTER"

    VIEW_IMAGE = "VIEW_IMAGE"

    MAIN_POSTER_TEXTURE = "MAIN_POSTER_TEXTURE"
    MAIN_POSTER_IMAGE = "MAIN_POSTER_IMAGE"

    DATA_TEX_BASE_NAME = "DATA_TEX_BASE_NAME"
    DATA_TEX_OVER_NAME = "DATA_TEX_OVER_NAME"
    DATA_TEX_RAW_NAME = "DATA_TEX_RAW_NAME"
    DATA_TEX_PROCESS_NAME = "DATA_TEX_PROCESS_NAME"
    DATA_TEX_REG_NAME = "DATA_TEX_REG_NAME"
    DATA_RESULT_DEF = "DATA_RESULT_DEF"
    DATA_CHECK_ANGLE_TYPE = "DATA_CHECK_ANGLE_TYPE"
    DATA_CHECK_NEED_SAVE = "DATA_CHECK_NEED_SAVE"
    DATA_CHECK_NEED_RAW_PROCESS = "DATA_CHECK_NEED_RAW_PROCESS"
    DATA_CHECK_DEBUG = "DATA_CHECK_DEBUG"

    INPUT_RASTER_SET_ANGLE = "INPUT_RASTER_SET_ANGLE"
    INPUT_RASTER_SET_DISTANCE = "INPUT_RASTER_SET_DISTANCE"
    INPUT_RASTER_SET_THICK = "INPUT_RASTER_SET_THICK"
    INPUT_RASTER_SET_OFFSET = "INPUT_RASTER_SET_OFFSET"
    INPUT_RASTER_DOUBLE_ANGLE = "INPUT_RASTER_DOUBLE_ANGLE"
    INPUT_RASTER_DOUBLE_OFFSET = "INPUT_RASTER_DOUBLE_OFFSET"
    INPUT_RASTER_DOUBLE_THICK = "INPUT_RASTER_DOUBLE_THICK"
    INPUT_RASTER_DOUBLE_DISTANCE = "INPUT_RASTER_DOUBLE_DISTANCE"
    INPUT_PROCESSOR_THRES_VALUE = "INPUT_PROCESSOR_THRES_VALUE"

    HANDLER_REG = "HANDLER_REG"
    MAIN_WIN_HANDLER = "MAIN_WIN_HANDLER"
    MAKE_RAW_IMAGE_HANDLER = "MAKE_RAW_IMAGE_HANDLER"


_win_dims = {Tag.WIN_STAT: WindowSettings(300, 468),
             Tag.WIN_VIEW: WindowSettings(300, 300),
             Tag.WIN_INPUT: WindowSettings(374, 768),
             Tag.WIN_CONTROL: WindowSettings(350, 768),
             Tag.WIN_MAIN_VIEW: WindowSettings(1024, 768)}

_win_labels = {Tag.WIN_STAT: "Data",
               Tag.WIN_VIEW: "Preview",
               Tag.WIN_INPUT: "Input",
               Tag.WIN_CONTROL: "Commands",
               Tag.WIN_MAIN_VIEW: "Main View"}

_win_pos = {Tag.WIN_STAT: (0, 0),
            Tag.WIN_VIEW: (0, 468),
            Tag.WIN_INPUT: (300, 0),
            Tag.WIN_CONTROL: (674, 0),
            Tag.WIN_MAIN_VIEW: (0, 0)}

_btn_labels = {Tag.BTN_INPUT_LOAD_BASE: "Load Base",
               Tag.BTN_INPUT_LOAD_OVER: "Load Over",
               Tag.BTN_INPUT_LOAD_RAW: "Load Raw",
               Tag.BTN_INPUT_LOAD_PROCESS: "Load Processed",
               Tag.BTN_INPUT_SHOW_RASTER: "Show Raster",
               Tag.BTN_CONTROL_MAKE_RAW: "Make Raw Image",
               Tag.BTN_CONTROL_ANALYSIS: "Start Analysis",
               Tag.BTN_CONTROL_SHOW_ANALYSIS: "Show Analysis Result",
               Tag.BTN_DATA_SHOW_CAMERA: "Show Camera",
               Tag.BTN_INPUT_RASTER_FACTORY: "Create Rasters",
               Tag.BTN_MAKE_RAW_PROCESS: "Process Raw Image"}

_inp_labels = {Tag.INPUT_RASTER_SET_ANGLE: "BRaster angle",
               Tag.INPUT_RASTER_SET_DISTANCE: "BRaster distance",
               Tag.INPUT_RASTER_SET_THICK: "BRaster line thick",
               Tag.INPUT_RASTER_DOUBLE_ANGLE: "DRaster angle",
               Tag.INPUT_RASTER_DOUBLE_OFFSET: "DRaster offset",
               Tag.INPUT_RASTER_DOUBLE_THICK: "DRaster line thick",
               Tag.INPUT_RASTER_DOUBLE_DISTANCE: "DRaster distance",
               Tag.INPUT_PROCESSOR_THRES_VALUE: "Set threshold"}


@dataclass
class DpgImageData:
    width: int
    height: int
    channels: Any
    data: Any


class TextureInstrument:
    def _process_poster_to_dpg(self, poster: ndarray) -> DpgImageData:
        texture_data = []
        width = poster.shape[0]
        height = poster.shape[1]

        for i in range(width):
            for j in range(height):
                texture_data.append(poster[i, j, 2])
                texture_data.append(poster[i, j, 1])
                texture_data.append(poster[i, j, 0])
                texture_data.append(255)

        return DpgImageData(width, height, None, texture_data)

    def paste_texture(self, texture_tag, dpg_data: Optional[DpgImageData] = None, poster: Optional[ndarray] = None):
        if dpg_data is None and poster is None:
            return False

        if dpg.does_item_exist(texture_tag):
            dpg.delete_item(texture_tag)

        if not dpg_data:
            dpg_data = self._process_poster_to_dpg(poster)

        dpg.add_static_texture(dpg_data.width, dpg_data.height,
                               dpg_data.data, tag=texture_tag, parent=Tag.TEXTURE_REG)
        return True

    def paste_image(self, texture_tag, on_view=True):
        if not dpg.does_item_exist(texture_tag):
            return False

        win_tag = Tag.WIN_VIEW if on_view else Tag.WIN_MAIN_VIEW
        image_tag = Tag.VIEW_IMAGE if on_view else Tag.MAIN_POSTER_IMAGE

        if dpg.does_item_exist(image_tag):
            dpg.delete_item(image_tag)

        dpg.add_image(texture_tag=texture_tag, tag=image_tag, parent=win_tag)
        return True


class Storage(TextureInstrument):
    def __init__(self):
        self._last_dict = {}
        self._objects = {}
        self._analyzator: Optional[Analizator] = None
        self._main_view_used = False
        self.distance_thick_min_diff = 5

    def callback(self, sender, app_data, user_data):
        print(sender)
        print(app_data)
        print(user_data)

    def load(self, sender, app_data, type_tag):
        def _set_texture_name():
            tag_dict = {Tag.TEXTURE_BASE: Tag.DATA_TEX_BASE_NAME,
                        Tag.TEXTURE_OVER: Tag.DATA_TEX_OVER_NAME,
                        Tag.TEXTURE_RAW: Tag.DATA_TEX_RAW_NAME,
                        Tag.TEXTURE_PROCESS: Tag.DATA_TEX_PROCESS_NAME}
            dpg.configure_item(
                item=tag_dict[type_tag], default_value=file_name)

        self._last_dict = app_data
        path = app_data["file_path_name"]
        file_name = app_data["file_name"][:70]

        width, height, channels, data = dpg.load_image(path)
        dpg_image_data = DpgImageData(width, height, channels, data)

        texture_to_data_tag = {Tag.TEXTURE_BASE: "raster",
                               Tag.TEXTURE_OVER: "raster",
                               Tag.TEXTURE_RAW: "raw",
                               Tag.TEXTURE_PROCESS: "process"}

        self._objects[type_tag] = api.load_image_by_tag(
            path, texture_to_data_tag[type_tag])
        self.paste_texture(type_tag, dpg_data=dpg_image_data)
        self.paste_image(type_tag, on_view=True)
        _set_texture_name()

    def make_raw_image(self, sender, app_data, user_data):
        if not dpg.does_item_exist(Tag.TEXTURE_BASE):
            return

        self.paste_image(Tag.TEXTURE_BASE, on_view=False)
        dpg.configure_item(Tag.WIN_VIEW, show=False)
        dpg.configure_item(Tag.WIN_MAIN_VIEW, show=True)

    def process_analysis(self, sender, app_data, user_data):
        base = self._objects.get(Tag.TEXTURE_BASE)
        over = self._objects.get(Tag.TEXTURE_OVER)
        process = self._objects.get(Tag.TEXTURE_PROCESS)

        if not all([base, over, process]):
            return

        self._analyzator = Analizator(base, over, process)

    def show_analysis_poster(self, sender, app_data, user_data):
        if not self._analyzator:
            return
        poster = self._analyzator.poster(select_persentile90=True)
        result = self._analyzator.has_deform()
        dpg.configure_item(
            Tag.DATA_RESULT_DEF, default_value=BY_DEFORM_MSG.get(result, "No defects"))
        api.imshow(poster)

    def camera_stream(self, sender, app_data, user_data):
        api.camera()
        image_data = api.get_picture()
        api.imshow(image_data.image)
        api.camera()

    def raster_factory(self, sender, app_data, user_data):
        tags = (Tag.INPUT_RASTER_SET_DISTANCE,
                Tag.INPUT_RASTER_SET_THICK, Tag.INPUT_RASTER_DOUBLE_ANGLE)
        angle = 0
        distance, thick, add_angle = dpg.get_values(tags)
        if math.fabs(distance - thick) < self.distance_thick_min_diff:
            return
        raster_base_settings = api.raster_settings(angle, distance, thick)
        raster_over_settings = api.raster_settings_double(
            raster_base_settings, add_angle=add_angle)

        need_save = dpg.get_value(Tag.DATA_CHECK_NEED_SAVE)
        base_raster = api.create_raster(
            _win_dims[Tag.WIN_MAIN_VIEW], raster_base_settings, need_save)
        over_raster = api.create_raster(
            _win_dims[Tag.WIN_MAIN_VIEW], raster_over_settings, need_save)

        if dpg.get_value(Tag.DATA_CHECK_DEBUG):
            api.imshow(base_raster.image)
            api.imshow(over_raster.image)

    def settings_filter(self, sender, app_data, user_data):
        value = dpg.get_value(sender)

        if sender == Tag.INPUT_RASTER_SET_THICK:
            distance = dpg.get_value(Tag.INPUT_RASTER_SET_DISTANCE)
            if value + self.distance_thick_min_diff >= distance:
                dpg.set_value(sender, value=distance -
                              self.distance_thick_min_diff)

        if sender == Tag.INPUT_RASTER_SET_DISTANCE:
            thick = dpg.get_value(Tag.INPUT_RASTER_SET_THICK)
            if value - self.distance_thick_min_diff <= thick:
                dpg.set_value(sender, value=thick +
                              self.distance_thick_min_diff)

        item_configs = dpg.get_item_configuration(sender)
        if item_configs["min_value"] > value:
            dpg.set_value(sender, value=item_configs["min_value"])
        if item_configs["max_value"] < value:
            dpg.set_value(sender, value=item_configs["max_value"])

    def process_raw_image(self, sender, app_data, user_data):
        raw_picture = self._objects.get(Tag.TEXTURE_RAW)
        if not raw_picture:
            return

        top_offset = 16
        threshold_value = dpg.get_value(Tag.INPUT_PROCESSOR_THRES_VALUE)
        processed_image = api.processor_pipeline(raw_picture, threshold_value,
                                                 top_offset, _win_dims[Tag.WIN_MAIN_VIEW])

        if dpg.get_value(Tag.DATA_CHECK_NEED_SAVE):
            saved_path = api.save_camera_image(processed_image.image)
            app_data_body = {"file_path_name": saved_path["to_camera"],
                             "file_name": saved_path["to_camera_filename"],
                             "_INNER_CALL": True}
            self.load(None, app_data_body, Tag.TEXTURE_PROCESS)

        if dpg.get_value(Tag.DATA_CHECK_DEBUG):
            api.imshow(processed_image.image)

    def key_q_pressed(self, sender, app_data, user_data):
        if self._main_view_used:
            return
        dpg.configure_item(Tag.WIN_MAIN_VIEW, show=False)
        dpg.configure_item(Tag.WIN_VIEW, show=True)

    def key_r_pressed(self, sender, app_data, user_data):
        main_view_config = dpg.get_item_configuration(Tag.WIN_MAIN_VIEW)
        if not main_view_config['show']:
            return

        self._main_view_used = True

        api.camera()
        raw_picture = api.get_picture()
        api.camera()

        self._main_view_used = False

        if dpg.get_value(Tag.DATA_CHECK_NEED_SAVE):
            saved_path = api.save_camera_image(raw_picture.image)
            app_data_body = {"file_path_name": saved_path["to_camera"],
                             "file_name": saved_path["to_camera_filename"],
                             "_INNER_CALL": True}
            self.load(None, app_data_body, Tag.TEXTURE_RAW)

        if dpg.get_value(Tag.DATA_CHECK_DEBUG):
            api.imshow(raw_picture.image)

        if dpg.get_value(Tag.DATA_CHECK_NEED_RAW_PROCESS):
            self.process_raw_image(None, None, None)

    def double_raster_type_changed(self, sender, app_data, user_data):
        value = dpg.get_value(sender)
        if value:
            dpg.configure_item(
                item=Tag.INPUT_RASTER_DOUBLE_ANGLE, enabled=True)
            dpg.configure_item(
                item=Tag.INPUT_RASTER_DOUBLE_OFFSET, enabled=False)
            dpg.configure_item(
                item=Tag.INPUT_RASTER_DOUBLE_THICK, enabled=False)
            dpg.configure_item(
                item=Tag.INPUT_RASTER_DOUBLE_DISTANCE, enabled=False)
        else:
            dpg.configure_item(
                item=Tag.INPUT_RASTER_DOUBLE_ANGLE, enabled=False)
            dpg.configure_item(
                item=Tag.INPUT_RASTER_DOUBLE_OFFSET, enabled=True)
            dpg.configure_item(
                item=Tag.INPUT_RASTER_DOUBLE_THICK, enabled=True)
            dpg.configure_item(
                item=Tag.INPUT_RASTER_DOUBLE_DISTANCE, enabled=True)


class Provider(Storage):
    def __init__(self):
        super().__init__()

    def construct(self):
        self._construct_windows()
        self._construct_registers()
        self._construct_dialog_windows()
        self._construct_buttons()
        self._construct_inputs()

    def _construct_buttons(self):
        group_input_tag = Tag.GROUP_INPUT_BTN_COLLECT
        dpg.add_group(tag=group_input_tag, horizontal=False,
                      parent=Tag.WIN_INPUT, width=300)

        btn_lbase = Tag.BTN_INPUT_LOAD_BASE
        dpg.add_button(tag=btn_lbase, label=_btn_labels[btn_lbase], parent=group_input_tag,
                       callback=lambda: dpg.show_item(Tag.FILE_DIALOG_LOAD_BASE))

        btn_lover = Tag.BTN_INPUT_LOAD_OVER
        dpg.add_button(tag=btn_lover, label=_btn_labels[btn_lover], parent=group_input_tag,
                       callback=lambda: dpg.show_item(Tag.FILE_DIALOG_LOAD_OVER))

        btn_lraw = Tag.BTN_INPUT_LOAD_RAW
        dpg.add_button(tag=btn_lraw, label=_btn_labels[btn_lraw], parent=group_input_tag,
                       callback=lambda: dpg.show_item(Tag.FILE_DIALOG_LOAD_RAW))

        btn_lprocess = Tag.BTN_INPUT_LOAD_PROCESS
        dpg.add_button(tag=btn_lprocess, label=_btn_labels[btn_lprocess], parent=group_input_tag,
                       callback=lambda: dpg.show_item(Tag.FILE_DIALOG_LOAD_PROCESS))

        btn_show_camera_tag = Tag.BTN_DATA_SHOW_CAMERA
        dpg.add_button(tag=btn_show_camera_tag, label=_btn_labels[btn_show_camera_tag], parent=group_input_tag,
                       callback=self.camera_stream)

        group_inputs_tag = Tag.GROUP_INPUT_INPUTS_COLLECT
        dpg.add_group(tag=group_inputs_tag, horizontal=False,
                      parent=Tag.WIN_INPUT, width=200)

        btn_raster_factory_tag = Tag.BTN_INPUT_RASTER_FACTORY
        dpg.add_button(tag=btn_raster_factory_tag, label=_btn_labels[btn_raster_factory_tag], parent=group_inputs_tag,
                       callback=self.raster_factory)

        group_control_tag = Tag.GROUP_CONTROL_BTN_COLLECT
        dpg.add_group(tag=group_control_tag, horizontal=False,
                      parent=Tag.WIN_CONTROL, width=200)

        btn_make_raw_raster = Tag.BTN_CONTROL_MAKE_RAW
        dpg.add_button(tag=btn_make_raw_raster, label=_btn_labels[btn_make_raw_raster], parent=group_control_tag,
                       callback=self.make_raw_image)

        btn_start_analysis = Tag.BTN_CONTROL_ANALYSIS
        dpg.add_button(tag=btn_start_analysis, label=_btn_labels[btn_start_analysis], parent=group_control_tag,
                       callback=self.process_analysis)

        btn_show_analysis = Tag.BTN_CONTROL_SHOW_ANALYSIS
        dpg.add_button(tag=btn_show_analysis, label=_btn_labels[btn_show_analysis], parent=group_control_tag,
                       callback=self.show_analysis_poster, user_data=Tag.TEXTURE_ANALIZATOR_POSTER)

        group_controls_tag = Tag.GROUP_CONTROL_PROCESS_COLLECT
        dpg.add_group(tag=group_controls_tag, horizontal=False,
                      parent=Tag.WIN_CONTROL, width=200)

        btn_raw_process_tag = Tag.BTN_MAKE_RAW_PROCESS
        dpg.add_button(tag=btn_raw_process_tag, label=_btn_labels[btn_raw_process_tag], parent=group_controls_tag,
                       callback=self.process_raw_image)

    def _construct_inputs(self):
        group_load_data_tag = Tag.GROUP_DATA_TEXT_COLLECTOR
        dpg.add_group(tag=group_load_data_tag, horizontal=False,
                      parent=Tag.WIN_STAT, width=200)

        dpg.add_input_text(tag=Tag.DATA_TEX_BASE_NAME, parent=group_load_data_tag,
                           default_value="...", label="Base name",
                           show=True, enabled=False)
        dpg.add_input_text(tag=Tag.DATA_TEX_OVER_NAME, parent=group_load_data_tag,
                           default_value="...", label="Over name",
                           show=True, enabled=False)
        dpg.add_input_text(tag=Tag.DATA_TEX_RAW_NAME, parent=group_load_data_tag,
                           default_value="...", label="Raw name",
                           show=True, enabled=False)
        dpg.add_input_text(tag=Tag.DATA_TEX_PROCESS_NAME, parent=group_load_data_tag,
                           default_value="...", label="Process name",
                           show=True, enabled=False)

        dpg.add_input_text(tag=Tag.DATA_RESULT_DEF, parent=group_load_data_tag,
                           default_value="...", label="Result",
                           show=True, enabled=False)

        dpg.add_checkbox(tag=Tag.DATA_CHECK_ANGLE_TYPE, label="Double Raster Angle Type",
                         parent=group_load_data_tag, callback=self.double_raster_type_changed,
                         default_value=True, show=False)
        dpg.add_checkbox(tag=Tag.DATA_CHECK_NEED_SAVE, label="Need Save", parent=group_load_data_tag,
                         default_value=False)
        dpg.add_checkbox(tag=Tag.DATA_CHECK_NEED_RAW_PROCESS, label="Need Raw Process", parent=group_load_data_tag,
                         default_value=False)
        dpg.add_checkbox(tag=Tag.DATA_CHECK_DEBUG, label="Debug", parent=group_load_data_tag,
                         default_value=False, show=False)

        group_raster_input_tag = Tag.GROUP_INPUT_INPUTS_COLLECT

        # angle_tag = Tag.INPUT_RASTER_SET_ANGLE
        # dpg.add_drag_int(tag=angle_tag, label=_inp_labels[angle_tag], parent=group_raster_input_tag,
        #                  min_value=0, max_value=180, default_value=0, callback=self.settings_filter)

        distance_tag = Tag.INPUT_RASTER_SET_DISTANCE
        dpg.add_drag_int(tag=distance_tag, label=_inp_labels[distance_tag], parent=group_raster_input_tag,
                         min_value=8, max_value=100, default_value=20, callback=self.settings_filter)

        thick_tag = Tag.INPUT_RASTER_SET_THICK
        dpg.add_drag_int(tag=thick_tag, label=_inp_labels[thick_tag], parent=group_raster_input_tag,
                         min_value=2, max_value=20, default_value=4, callback=self.settings_filter)

        double_angle_tag = Tag.INPUT_RASTER_DOUBLE_ANGLE
        dpg.add_drag_int(tag=double_angle_tag, label=_inp_labels[double_angle_tag], parent=group_raster_input_tag,
                         min_value=30, max_value=60, default_value=45, callback=self.settings_filter)

        # distance_tag = Tag.INPUT_RASTER_DOUBLE_DISTANCE
        # dpg.add_drag_int(tag=distance_tag, label=_inp_labels[distance_tag], parent=group_raster_input_tag,
        #                  min_value=8, max_value=100, default_value=20, callback=self.settings_filter,
        #                  enabled=False)
        #
        # thick_tag = Tag.INPUT_RASTER_DOUBLE_THICK
        # dpg.add_drag_int(tag=thick_tag, label=_inp_labels[thick_tag], parent=group_raster_input_tag,
        #                  min_value=2, max_value=20, default_value=4, callback=self.settings_filter,
        #                  enabled=False)
        #
        # double_offset_tag = Tag.INPUT_RASTER_DOUBLE_OFFSET
        # dpg.add_drag_int(tag=double_offset_tag, label=_inp_labels[double_offset_tag], parent=group_raster_input_tag,
        #                  min_value=0, max_value=100, default_value=0, callback=self.settings_filter,
        #                  enabled=False)

        # =======================================================================================

        group_control_process_tag = Tag.GROUP_CONTROL_PROCESS_COLLECT

        thres_input_tag = Tag.INPUT_PROCESSOR_THRES_VALUE
        dpg.add_drag_int(tag=thres_input_tag, label=_inp_labels[thres_input_tag], parent=group_control_process_tag,
                         min_value=10, max_value=250, default_value=100, callback=self.settings_filter)

    def _construct_registers(self):
        dpg.add_texture_registry(tag=Tag.TEXTURE_REG)
        dpg.add_handler_registry(tag=Tag.HANDLER_REG)
        dpg.add_key_press_handler(tag=Tag.MAIN_WIN_HANDLER, parent=Tag.HANDLER_REG,
                                  key=dpg.mvKey_Q, callback=self.key_q_pressed)
        dpg.add_key_press_handler(tag=Tag.MAKE_RAW_IMAGE_HANDLER, parent=Tag.HANDLER_REG,
                                  key=dpg.mvKey_R, callback=self.key_r_pressed)

    def _construct_windows(self):
        stat_tag = Tag.WIN_STAT
        dpg.add_window(tag=stat_tag, label=_win_labels[stat_tag], pos=_win_pos[stat_tag],
                       width=_win_dims[stat_tag].width, height=_win_dims[stat_tag].height,
                       no_move=True, no_resize=True, no_collapse=True,
                       no_close=True, no_scrollbar=True)

        view_tag = Tag.WIN_VIEW
        dpg.add_window(tag=view_tag, label=_win_labels[view_tag], pos=_win_pos[view_tag],
                       width=_win_dims[view_tag].width, height=_win_dims[view_tag].height,
                       no_move=True, no_resize=True, no_collapse=True,
                       no_close=True, no_scrollbar=True)

        input_tag = Tag.WIN_INPUT
        dpg.add_window(tag=input_tag, label=_win_labels[input_tag], pos=_win_pos[input_tag],
                       width=_win_dims[input_tag].width, height=_win_dims[input_tag].height,
                       no_move=True, no_resize=True, no_collapse=True,
                       no_close=True, no_scrollbar=True)

        control_tag = Tag.WIN_CONTROL
        dpg.add_window(tag=control_tag, label=_win_labels[control_tag], pos=_win_pos[control_tag],
                       width=_win_dims[control_tag].width, height=_win_dims[control_tag].height,
                       no_move=True, no_resize=True, no_collapse=True,
                       no_close=True, no_scrollbar=True)

        analysis_view_tag = Tag.WIN_MAIN_VIEW
        width, height = _win_dims[analysis_view_tag].width, _win_dims[analysis_view_tag].height
        dpg.add_window(tag=analysis_view_tag, label=_win_labels[analysis_view_tag], pos=_win_pos[analysis_view_tag],
                       width=width, height=height, on_close=lambda: dpg.configure_item(
                           analysis_view_tag, show=False),
                       show=False, no_scrollbar=True, no_resize=True, no_move=True, no_title_bar=True, autosize=True)

    def _construct_dialog_windows(self):
        file_label = "File Dialog"
        file_width = 800
        file_height = 600
        default_path = "D:\\"

        file_lbase = Tag.FILE_DIALOG_LOAD_BASE
        dpg.add_file_dialog(tag=file_lbase, label=file_label, min_size=(file_width, file_height),
                            width=file_width, height=file_height, callback=self.load, user_data=Tag.TEXTURE_BASE,
                            show=False, directory_selector=False, default_path=default_path)

        file_lover = Tag.FILE_DIALOG_LOAD_OVER
        dpg.add_file_dialog(tag=file_lover, label=file_label, min_size=(file_width, file_height),
                            width=file_width, height=file_height, callback=self.load, user_data=Tag.TEXTURE_OVER,
                            show=False, directory_selector=False, default_path=default_path)

        file_lraw = Tag.FILE_DIALOG_LOAD_RAW
        dpg.add_file_dialog(tag=file_lraw, label=file_label, min_size=(file_width, file_height),
                            width=file_width, height=file_height, callback=self.load, user_data=Tag.TEXTURE_RAW,
                            show=False, directory_selector=False, default_path=default_path)

        file_lprocess = Tag.FILE_DIALOG_LOAD_PROCESS
        dpg.add_file_dialog(tag=file_lprocess, label=file_label, min_size=(file_width, file_height),
                            width=file_width, height=file_height, callback=self.load, user_data=Tag.TEXTURE_PROCESS,
                            show=False, directory_selector=False, default_path=default_path)

        dpg.add_file_extension(extension=".png", color=(
            150, 255, 150, 255), parent=file_lbase)
        dpg.add_file_extension(extension=".png", color=(
            150, 255, 150, 255), parent=file_lover)
        dpg.add_file_extension(extension=".png", color=(
            150, 255, 150, 255), parent=file_lraw)
        dpg.add_file_extension(extension=".png", color=(
            150, 255, 150, 255), parent=file_lprocess)


class Interface:
    def __init__(self, v_width=1024, v_height=768, v_title="Viewport"):
        self._viewport_width = v_width
        self._viewport_height = v_height
        self._viewport_title = v_title
        self.provider = None

    def _apply(self):
        self.provider = Provider()
        self.provider.construct()

    def start(self):
        dpg.create_context()
        dpg.create_viewport(title=self._viewport_title, width=self._viewport_width, height=self._viewport_height,
                            x_pos=0, y_pos=0)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        self._apply()

        dpg.start_dearpygui()
        dpg.destroy_context()

    def start_demo(self):
        dpg.create_context()
        dpg.create_viewport(title=self._viewport_title,
                            width=self._viewport_width, height=self._viewport_height)

        demo.show_demo()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
