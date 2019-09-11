'''
Class that reads JSON and puts it into a dictionary of appropriate format for use by other classes
'''
import json


class JsonReader:
    def __init__(self):
        print("todo: JsonReader init")

    def read(self, path):
        file = open(path + "_output.txt") #TODO: determine how much of a path will be passed in
        json_file = file.read()
        output_dictionary = json.loads(json_file) #get the dictionary
        return output_dictionary
