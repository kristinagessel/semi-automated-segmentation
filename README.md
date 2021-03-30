# semi-automated-segmentation
This project was created by Kristina Gessel at the University of Kentucky as a Master's project. (2019-2020)
This code was written as a proof of concept. It has been fully deprecated in favor of a cleaner and significantly faster C++ implementation that is part of EduceLab's Volume Cartographer library (https://gitlab.com/educelab/volume-cartographer)

The repository contains several helper scripts. 
This project investigates supervised machine learning methods for volumetric segmentation of a manuscript as well, so there are simple HDF5 tools for converting a labeled subvolume into HDF5 for the tested neural network.
There are tools to convert the segmentations we create with ExtrapolateMask (see below) into a point cloud to be viewed and textured in 3D.

The primary script for semi-automated segmentation is ExtrapolateMask.
ExtrapolateMask segments individual pages in a micro-CT scan of a manuscript. It requires an initial set of points specified by a user, which can be passed in as a simple .txt file or from EduceLab's Volume Cartographer GUI.

The full project report can be found here: https://docs.google.com/document/d/1fOpQ29GhEHLT3VVXmVYhUPK-1RFpLviFVw5fBkib9LQ/edit?usp=sharing
