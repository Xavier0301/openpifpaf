import openpifpaf

from . import datamodule


def register():
    openpifpaf.DATAMODULES['vispanel'] = datamodule.VisPanelModule
