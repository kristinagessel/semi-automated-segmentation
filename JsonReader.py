'''
Class that reads JSON and puts it into a dictionary of appropriate format for use by other classes
'''
import json


class JsonReader:
    def __init__(self):
        self.file = ""

    def read(self, path, page_num, extra_end_txt):
        file = open(path + page_num + extra_end_txt)
        output_dictionary = json.loads(file.read()) #get the dictionary
        return output_dictionary
