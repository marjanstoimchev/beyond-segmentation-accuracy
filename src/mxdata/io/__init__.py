"""Slide IO: file discovery, CellDIVE channel names, level-0 boxes, and the lazy pyramid reader."""
from .channels import ChannelInfo, parse_channel_filename
from .geometry import Box
from .paths import DATA_ROOT, DEFAULT_DATASET, SlidePaths
from .slide import MxSlide

__all__ = ["DATA_ROOT", "DEFAULT_DATASET", "SlidePaths", "ChannelInfo", "parse_channel_filename", "Box", "MxSlide"]
