import math
import os
import re
import struct

import numpy as np
import progressbar


class VCPSReader:
    def __init__(self, path):
        self._path = path
        self.point_positions = {}

        ppm_path_stem, _ = os.path.splitext(self._path)

    @staticmethod
    def parse_VCPS_header(filename):
        #These regexes stay the same with VCPS--same header
        comments_re = re.compile('^#')
        width_re = re.compile('^width')
        height_re = re.compile('^height')
        dim_re = re.compile('^dim')
        ordering_re = re.compile('^ordered')
        type_re = re.compile('^type')
        version_re = re.compile('^version')
        header_terminator_re = re.compile('^<>$')

        with open(filename, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8')
                if comments_re.match(line):
                    pass
                elif width_re.match(line):
                    width = int(line.split(': ')[1])
                elif height_re.match(line):
                    height = int(line.split(': ')[1])
                elif dim_re.match(line):
                    dim = int(line.split(': ')[1])
                elif ordering_re.match(line):
                    ordering = line.split(': ')[1].strip() == 'true'
                elif type_re.match(line):
                    val_type = line.split(': ')[1].strip()
                    assert val_type in ['double']
                elif version_re.match(line):
                    version = line.split(': ')[1].strip()
                elif header_terminator_re.match(line):
                    break
                else:
                    print('Warning: VCPS header contains unknown line: {}'.format(line.strip()))

        return {
            'width': width,
            'height': height,
            'dim': dim,
            'ordering': ordering,
            'type': val_type,
            'version': version
        }

    def process_VCPS_file(self, dict):
        """Read a VCPS file and store the data in the VCPS object.
        """
        filename = self._path
        if any(dict): #If the dictionary contains anything, start with it
            self.point_positions.update(dict)

        header = VCPSReader.parse_VCPS_header(filename)
        self._width = header['width']
        self._height = header['height']
        self._dim = header['dim']
        self._ordering = header['ordering']
        self._type = header['type']
        self._version = header['version']

        print(
            'Processing VCPS data for {} with width {}, height {}, dim {}... '.format(
                self._path, self._width, self._height, self._dim
            )
        )

        self._data = np.empty((self._height, self._width, self._dim))

        with open(filename, 'rb') as f:
            header_terminator_re = re.compile('^<>$')
            while True:
                line = f.readline().decode('utf-8')
                if header_terminator_re.match(line):
                    break

            bar = progressbar.ProgressBar()
            dims = ["x", "y", "z-slice"]
            for y in bar(range(self._height)):
                for x in range(self._width):
                    position_values = []
                    for idx in range(self._dim):
                            self._data[y, x, idx] = struct.unpack('d', f.read(8))[0]
                            position_values.append(self._data[y, x, idx])
                    #Add a tuple of (x,y) to the dict with the correct slice number (position_values has: [x, y, z(slice)]
                    self.add_to_dict(position_values[0], position_values[1], position_values[2])
        return self.point_positions


    def add_to_dict(self, x, y, slice):
        if math.floor(slice) not in self.point_positions:
            self.point_positions[math.floor(slice)] = [] #init array so it's happy
        self.point_positions[math.floor(slice)].append((x, y))
