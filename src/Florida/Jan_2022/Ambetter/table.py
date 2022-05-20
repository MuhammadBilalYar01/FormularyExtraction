class Table:
    def __init__(self, bbox, vlines, hlines):
        self.bbox = bbox
        self.table_settings = {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": vlines,
        "explicit_horizontal_lines": hlines,
        "keep_blank_chars": True,
        }
        # self.table_settings["explicit_vertical_lines"] = vlines
        # self.table_settings["explicit_horizontal_lines"] = hlines

    def set_hlines(self, hlines):
        self.table_settings["explicit_horizontal_lines"] = hlines

    def update_hlines(self, hlines):
        # Extend the hlines list
        orig_hlines = self.table_settings["explicit_horizontal_lines"].copy()
        self.table_settings["explicit_horizontal_lines"] = orig_hlines + hlines         
        # self.table_settings["explicit_horizontal_lines"].extend(hlines)