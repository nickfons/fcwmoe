This is the concise version of the Face Completion with Multi-Estimator Output

To run the notebooks:
 - Start a venv, and run pip install -r requirements.txt
 - Launch jupyter notebook
 - Open project_home.ipynb and go from there

The full project directory is available on requested, but the directory currently sits at ~100 gigabytes of data,
so it will need to uploaded in some other way.

This directory includes the following:
 - example: folder with a subset of the CelebA dataset, and various other neccesities
  - inputs:
   - initialimageset: 100 sample subset of celebA
   - preprocessing: 5 extra images to be pre-processed
   - warpedimagesubset
    - images: pre-processed images
  - outputs/dcgan_results: contains past dcgan results
 - plurinpaint: an edited module version of https://github.com/lyndonzheng/Pluralistic-Inpainting
 - masks: a mask for plurinpain
 - venv: should hopefully be sourceable on all platforms
 - base_dcgan.py: python file implementing a dcgan, edited from the tutorial mentioned in its header
 - calculate_results.py: file to calculate MSEs and MAEs
 - canonical_face...: image of Google's mediaPipe face model
 - dask_fcmeo.py: implementation of all regressors, basically the main loop
 - estimators.ipynb: jupyter notebook with estimator information
 - pre_proccessing.ipynb: jupyter notebook with pre-processing information
 - preprocessing.py: an older pre-processing pipeline dealing with facial landmarks
 - project_home.ipynb: summary jupyter notebook
 - README.txt: this file
 - rgb_preprocessing.py: current pre-processing pipeline, implementing full pipeling in align_and_crop