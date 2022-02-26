##########################################################################################
#
# hyperspectralalignment: joint work between CNRS-AIST JRL, NAIST OMI lab and UPJV MIS lab
#
# Authors: Guillaume Caron, Takuya Funatomi
#
# Dates: from November 2021 to February 2022
##########################################################################################

0. Prerequisites to run these codes: 
     install cmake (version 3.16.3 tested)
     install ceres-solver (version 2.0.0, github, master, commit 8d3e64dd5e64b346ed3e412cba2b70d760881bfb tested)
     install opencv (version 4.2.0 tested, sources are required for RANSAC)
     install libnpy (github, master, commit b67016355374fe29d75430842d72865f99e3f697 tested)
     download test data from: http://mis.u-picardie.fr/~g-caron/data/2022_hsalign_media.zip and set its content in the 2022_hsalign_media directory in the hyperspectralalignment directory
1. create a new directory named build in the hyperspectralalignment directory
2. use cmake to fill the build directory in, with variables:
     OpenCV_SRC_DIR set to /your/path/to/opencv/source/dir (the directory in which the modules directory is, e.g. in command line: cmake .. -DCMAKE_BUILD_TYPE=Release -D OpenCV_SRC_DIR=/your/path/to/opencv/source/dir)
     libnpy_INCLUDE_DIRS set to the include directory of libnpy (e.g. adding -D libnpy_INCLUDE_DIRS=/your/path/to/libnpy/include/dir to the cmake command line)
3. open the project in build or use the make command in the latter directory to build the exe file
4. run the programs from the command line from the hyperspectralalignment directory, considering it includes the 2022_hsalign_media directory, with arguments as:
     
./build/multiSalign_sphericalHomog_hyperspectral2gray \
   -desired './2022_hsalign_media/XVIII_ThetaV_eighth/R0010622.jpg' \
   -input './2022_hsalign_media/XVIII_HSI/cor-0.npy' \
   -hsiazel './2022_hsalign_media/XVIII_HSI/angles-0.npy' \
   -hsi_wavelengths './2022_hsalign_media/XVIII_HSI/wavelengths.npy' \
   -output './2022_hsalign_media/XVIII_hyperspectral_cor-0_to_R0010622.jpg' \
   -thetaROI './2022_hsalign_media/XVIII_ThetaV_eighth/thetaROI.txt' \
   -outputMappingOpenCVTxt './2022_hsalign_media/XVIII_ThetaV_eighth/hsiIndicesInThetaFrame.txt' \
   -outputMappingNpy './2022_hsalign_media/XVIII_ThetaV_eighth/hsiIndicesInThetaFrame.npy' \
   -intensityFact 40 
   -directOpt
   -nodisplayUsedArea 

One can also add options to use corresponding points from text files, e.g. by adding:
   \
   -thetaKeypoints './2022_hsalign_media/XVIII_ThetaV_eighth/thetaKeypoints.txt' \
   -hsiKeypoints './2022_hsalign_media/XVIII_HSI/hsiTripodEquiKeypoints.txt'
