# semi-automated-segmentation
This project was created by Kristina Gessel at the University of Kentucky as a Master's project. (2019-2020)

The repository contains several helper scripts. 
This project investigates supervised machine learning methods for volumetric segmentation of a manuscript as well, so there are simple HDF5 tools for converting a labeled subvolume into HDF5 for the tested neural network.
There are tools to convert the segmentations we create with ExtrapolateMask (see below) into a point cloud to be viewed and textured in 3D.

The primary script for semi-automated segmentation is ExtrapolateMask (name may change in the future.)
ExtrapolateMask segments individual pages in a micro-CT scan of a manuscript. It requires an initial set of points specified by a user, which can be passed in as a simple .txt file or from the Digital Restoration Initiative's Volume Cartographer software.


