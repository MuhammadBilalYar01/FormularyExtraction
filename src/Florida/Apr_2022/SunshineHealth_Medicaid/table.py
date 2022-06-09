class Table:

    def __init__(self, bbox, vlines):
        self.bbox = bbox
        self.table_settings = {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "text",
        "explicit_vertical_lines":vlines,
        "intersection_x_tolerance": 15,
        "keep_blank_chars": True,
        }

    # def set_hlines(self, hlines):
    #     self.table_settings["explicit_horizontal_lines"] = hlines

    # def update_hlines(self, hlines):
    #     # Extend the hlines list
    #     orig_hlines = self.table_settings["explicit_horizontal_lines"].copy()
    #     self.table_settings["explicit_horizontal_lines"] = orig_hlines + hlines         
    #     # self.table_settings["explicit_horizontal_lines"].extend(hlines)

    # def set_hstrategy(self, hstrategy):
    #     self.table_settings["horizontal_strategy"] = hstrategy