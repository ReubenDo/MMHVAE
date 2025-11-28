# Repository to prepare the ReMIND dataset
Code to run the processing to co-register images from the ReMIND dataset.

## 1. Requirements

- Create your favorite Python environment and install the required packages.

``` pip install -r requirements.txt ```

- Download the `nrrd.zip` file [here](https://owncloud.icm-institute.org/index.php/s/i1VxgYJZqWcE2v1), move it to `./data/` and unzip it there.

## 2. Overview of the pre-computed registration transformations
<img width="960" height="430" alt="image" src="https://github.com/user-attachments/assets/286ab964-5a24-48d6-a3e2-5e5bba1115bc" />

## 3. Scripts to coregister the data
- Co-register all the data in the MRI space:

``` python create_dataset_mri-space.py ```

- Co-register all the data in the iUS space:

``` python create_dataset_us-space.py ```

