import h5py
import numpy as np

#path to file:
path = "/Users/kristinagessel/Downloads/h5pyfile.h5"#"/Users/kristinagessel/Desktop/ProjectExperiments/floodfill-network/ffn-github-repo/test_h5pyfile.h5"

out_file = open("point_cloud_test.txt", 'w')
f = h5py.File(path, 'r')
key_list = list(f.keys())
print(key_list)

dataset = f['seg']
print(dataset.shape)
print(dataset.dtype)

num_color_dict = {}
pg_colors = []
obj_limit = 500

#record anything that's not 0 through all the layers (shape[2] is hopefully z, verify)
for z in range(0, dataset.shape[2]):
    for y in range(0, dataset.shape[1]):
        for x in range(0, dataset.shape[0]):
            if dataset[x][y][z] != 0:
                # if object class has been seen before, use the same color
                if dataset[x][y][z] in num_color_dict.keys():
                    color = num_color_dict[dataset[x][y][z]]
                    out_file.write(str(x) + " " + str(y) + " " + str(z) + " " + str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + "\n")
                elif len(num_color_dict.keys()) < obj_limit:
                    color = tuple(np.random.choice(range(256), size=3))
                    if(len(num_color_dict.keys()) < 10):
                        while color in pg_colors:
                            color = tuple(np.random.choice(range(256), size=3))
                        pg_colors.append(color)
                        num_color_dict[dataset[x][y][z]] = color
                    out_file.write(str(x) + " " + str(y) + " " + str(z) + " " + str(color[0]) + " " + str(color[1]) + " " + str(color[2]) +"\n")

f.close()
out_file.close()