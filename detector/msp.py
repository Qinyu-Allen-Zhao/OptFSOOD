from .base import BasePostprocessor


class MSP(BasePostprocessor):
    def __init__(self):
        super().__init__()
        self.aps_mode = False
