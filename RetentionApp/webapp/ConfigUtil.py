import re

class ConfigUtil(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.config_params ={}
        self.file_parser()
        self.set_params()
        self.lines = []

    def file_parser(self):
        file_object = open(self.filepath, "r")
        self.lines = file_object.readlines()

    def add_params(self ,default_index):
        for line in self.lines:
            if ("#" not in line) and (line != "\n"):
                name_val_pair = line.split("=")
                self.config_params[name_val_pair[0]] = re.sub(r'\n', '', name_val_pair[1])
            elif "Default" in line:
                break
            default_index += 1
        return default_index

    def set_params(self):
        default_index =self.add_params(0)
        print(self.config_params)
        if(len(self.config_params )== 0):
            self.add_params(default_index + 1)

    def get_params(self):
        return self.config_params


