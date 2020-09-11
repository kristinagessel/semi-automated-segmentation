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
        self.val_type = 'double'

        ppm_path_stem, _ = os.path.splitext(self._path)

    @staticmethod
    def parse_unordered_VCPS_header(filename):
        comments_re = re.compile('^#')
        size_re = re.compile('^size')
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
                elif dim_re.match(line):
                    dim = int(line.split(': ')[1])
                elif ordering_re.match(line):
                    ordering = line.split(': ')[1].strip() == 'true'
                elif type_re.match(line):
                    val_type = line.split(': ')[1].strip()
                    assert val_type in ['double', 'int']
                elif version_re.match(line):
                    version = line.split(': ')[1].strip()
                elif size_re.match(line):
                    size = line.split(': ')[1].strip()
                elif header_terminator_re.match(line):
                    break
                else:
                    print('Warning: VCPS header contains unknown line: {}'.format(line.strip()))

        return {
            'size': size,
            'dim': dim,
            'ordering': ordering,
            'type': val_type,
            'version': version
        }

    @staticmethod
    def parse_VCPS_header(filename):
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
                    self.add_to_dict(position_values[0], position_values[1], position_values[2])
        return self.point_positions

    def process_unordered_VCPS_file(self, dict):
        """Read a VCPS file and store the data in the VCPS object.
        """
        filename = self._path
        if any(dict): #If the dictionary contains anything, start with it
            self.point_positions.update(dict)

        header = VCPSReader.parse_unordered_VCPS_header(filename)
        self._size = header['size']
        self._dim = header['dim']
        self._ordering = header['ordering']
        self._type = header['type']
        self._version = header['version']

        print(
            'Processing VCPS data for {} with dim {} and size {}... '.format(
                self._path, self._dim, self._size
            )
        )

        #self._data = np.empty((self._height, self._width, self._dim))

        with open(filename, 'rb') as f:
            header_terminator_re = re.compile('^<>$')
            while True:
                line = f.readline().decode('utf-8')
                if header_terminator_re.match(line):
                    break

            bar = progressbar.ProgressBar()
            position_values = []
            for idx in bar(range(int(self._size))):
                #need a case for an int or double value type depending on what we get back...
                if self._type in ['double']:
                    x = struct.unpack('d', f.read(8))[0]
                    y = struct.unpack('d', f.read(8))[0]
                    z = struct.unpack('d', f.read(8))[0]
                elif self._type in ['int']:
                    x = struct.unpack('i', f.read(4))[0]
                    y = struct.unpack('i', f.read(4))[0]
                    z = struct.unpack('i', f.read(4))[0]
                #print("(", x, ", ", y, ", ", z, ")")
                self.add_to_dict(x, y, z)
        return self.point_positions


    def add_to_dict(self, x, y, slice):
        if math.floor(slice) not in self.point_positions:
            self.point_positions[math.floor(slice)] = []
        self.point_positions[math.floor(slice)].append((x, y))
