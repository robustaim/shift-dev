<h1 align="center"> SHIFT Dataset DevKit </h1>

This repo contains tools and scripts for [SHIFT Dataset](https://www.vis.xyz/shift/)'s downloading, conversion, and more!

[**Homepage**](https://www.vis.xyz/shift) | [**Paper (CVPR 2022)**](https://arxiv.org/abs/2206.08367) | [**Poster**](https://github.com/SysCV/shift-dev/blob/main/assert/Poster%20SHIFT.pdf) | [**Talk**](https://www.youtube.com/watch?v=q39gJveIhRc) | [**Demo**](https://www.youtube.com/watch?v=BsqGrDd2Kzw)


<div align="center">
<div></div>

| **RGB**          |    **Optical Flow**    | **Depth**   | **LiDAR** |
|:----------------:|:----------------:|:----------------:|:---------:|
|  <img src="assert/figures/img.png">                |       <img src="assert/figures/flow.png">     |   <img src="assert/figures/depth.png">                       |   <img src="assert/figures/lidar.png" >         |
|   **Bounding box** | **Instance Segm.** | **Semantic Segm.**  | **Body Pose (soon)**  |
|   <img src="assert/figures/bbox2d.png">                 |     <img src="assert/figures/ins.png">            |         <img src="assert/figures/seg.png">           |       <img src="assert/figures/pose.png">      |

</div>



## News
- **[Jun 2023]** We are organizing [challenges](https://wvcl.vis.xyz/challenges) based on SHIFT at the [VCL Workshop](https://wvcl.vis.xyz/), ICCV 2023. Please come and win the prizes!
- **[Feb 2023]** Reference data loaders for PyTorch and mmdet are released! ([examples](https://github.com/SysCV/shift-dev/blob/main/examples))
- **[Sept 2022]** We released visualization scripts for annotation and sensor pose (issue https://github.com/SysCV/shift-dev/issues/6).
- **[June 2022]** We released the DevKit repo!


## Downloading
We recommend downloading SHIFT using our Python download script. You can select the subset of views, data groups, splits, and framerates of the data to download. A usage example is shown below. You can find the abbreviation for views and data groups in the following tables.

```bash
python download.py --view  "[front, left_stereo]" \   # list of view abbreviation to download
                   --group "[img, semseg]" \          # list of data group abbreviation to download 
                   --split "[train, val, test]" \     # list of splits to download 
                   --framerate "[images, videos]" \   # chooses the desired frame rate (images=1fps, videos=10fps)
                   --shift "discrete" \               # type of domain shifts. Options: discrete, continuous/1x, continuous/10x, continuous/100x 
                   dataset_root                       # path where to store the downloaded data
```
**Example 1**: Download the entire RGB images and 2D bounding boxes from the discrete shift data.
```bash
python download.py --view "all" --group "[img, det_2d]" --split "all" --framerate "[images]" ./data
```

**Example 2**: Download the entire front view images and all annotations from the discrete shift data.
```bash
python download.py --view "[front]" --group "all" --split "all" --framerate "[images]" ./data
```

### Manually download
You could find the download links on our [download page](https://www.vis.xyz/shift/download/) or [file server](https://dl.cv.ethz.ch/shift/).

## Data Loaders

We have implemented reference dataset classes for SHIFT. They
show how to load the data via the PyTorch Dataset ([torch_dataset.py](https://github.com/SysCV/shift-dev/blob/main/examples/torch_dataset.py)) and CustomDataset in [mmdet](https://github.com/open-mmlab/mmdetection) framework ([mmdet_dataset.py](https://github.com/SysCV/shift-dev/blob/main/examples/mmdet_dataset.py)).

Below is an example for PyTorch Dataset.

```python
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend

dataset = SHIFTDataset(
    data_root="./SHIFT_dataset/",
    split="train",
    keys_to_load=[
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
        Keys.segmentation_masks,
    ],
    views_to_load=["front"],
    framerate="images",
    shift_type="discrete",
    backend=ZipBackend(),  # also supports HDF5Backend(), FileBackend()
    verbose=True,
)
```



## Tools
<details>
<summary>
<h3>Packing Zip file into HDF5 </h3>
</summary>

Instead of unzipping the downloaded zip files, you can also convert them into corresponding [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files. HDF5 file is designed to store a large dataset in a single file and, meanwhile, to support efficient I/O for training purposes. Converting to HDF5 is a good practice in an environment where the number of files that can be stored is limited. 

However, if you want to preprocess the data before using them, we **don't recommend** converting them into HDF5 before the processing, which will complicate the loading.

**Example 1**: Packing directly from the downloaded zip files. (You can set the number of processes by `-j`)
```bash
python -m shift_dev.io.to_hdf5 "./data/discrete/**/*.zip" --zip -j 1
```

**Example 2**: Packing from an unzipped folder.
```bash
python -m shift_dev.io.to_hdf5 "./data/discrete/images/val/left_45/img/"
```

Note: The converted HDF5 file will maintain the same file structure of the zip file/folder, i.e., `<seq>/<frame>_<group>_<view>.<ext>`.
</details>

<details>
<summary>
<h3>Reading from HDF5 files</h3>
</summary>

Below is a code snippet for reading images from an HDF5 file.
```python
import io
import h5py
from PIL import Image

name = "0123-abcd/00000000_img_front.jpg"
with h5py.File("/path/to/file.hdf5", "r") as hdf5:      # load the HDF5 file
    bytes = bytearray(hdf5[name])                       # select the file we want
    img = Image.open(io.BytesIO(bytes))                 # same as opening an ordinary png file from IO stream.
```

Below is a code snippet for reading point clouds from an HDF5 file.
```python
import io
import h5py
import plyfile

name = "0123-abcd/00000000_lidar_center.ply"
bytes = io.BytesIO(np.array(hdf5[name]))              # create an IO buffer
plydata = plyfile.PlyData.read(bytes)                 # parse point cloud from the buffer

num_points = plydata['vertex'].count
arr = np.zeros((num_points, 4), dtype=np.float32)     # array of [n, 4], columns are: x, y, z, intensity
arr[:, 0] = plydata['vertex'].data['x']
arr[:, 1] = plydata['vertex'].data['y']
arr[:, 2] = plydata['vertex'].data['z']
arr[:, 3] = plydata['vertex'].data['intensity']
```

</details>

<details>
<summary>
<h3>Decompress video files</h3>
</summary>

For easier retrieval of frames during training, we recommend decompressing all video sequences into image frames before training. Make sure there is enough disk space to store the decompressed frames. The video sequences are used for the RGB data group only.

The mode option (`--mode, -m`) controls the storage type of the decompressed frames. When the mode is set to `folder` (default option) the frames are extracted to local file systems directly; when the mode is set to `zip`, `tar` or `hdf5`, the frames are stored in the corresponding archive file, e.g., `img_decompressed.zip`.  

All frames will be saved using the same name pattern of `<seq>/<frame>_<group>_<view>.<ext>`.

- To use your local FFmpeg libraries (4.x) is supported. You can follow the command example below, which decompresses videos to image frames and store them into a zip archive with the same filename as the tar file.
    ```bash
    python -m shift_dev.io.decompress_videos "discrete/videos/val/front/*.tar" -m "zip" -j 1
    ```

- To ensure reproducible decompression of videos, we recommend using our Docker image. You could refer to the Docker engine's [installation doc](https://docs.docker.com/engine/install/).
    ```bash
    # build and install our Docker image
    docker build -t shift_dataset_decompress .

    # run the container (the mode is set to "hdf5")
    docker run -v <path/to/data>:/data -e MODE=hdf5 shift_dataset_decompress
    ```
    Here, `<path/to/data>` denotes the root path under which all tar files will be processed recursively. The mode and number of jobs can be configured through environment variables `MODE` and `JOBS`. 
</details>

<details>
<summary>
<h3>Visualization</h3>
</summary>

We provide a visualization tool for object-level labels (e.g., bounding box, instance segmentation). The main rendering functions are provided in `shift_dev/vis/render.py` file. We believe you can reuse many of them for other kinds of visualization. 

We also provide a tool to make videos with annotations:
```bash
python -m shift_dev.vis.video <seq_id> \    # specify the video sequence
    -d <path/to/img.zip> \                  # path to the img.zip or its unzipped folder
    -l <path/to/label.json> \               # path to the corresponding label ({det_2d/det_3d/det_insseg_2d}.json)
    -o <path/for/output> \                  # output path
    --view front                            # specify the view, needed to be corresponded with images and label file
```
This command will render an MP4 video with the bounding boxes or instance masks plotted over the background images. Check out the example [here](https://www.youtube.com/watch?v=BsqGrDd2Kzw) (starting from 00:10)!
</details>


## Resources

You can find some helpful repos based on the SHIFT dataset below! Please don't hesitate to contact us to include your repos.
- **Test-time adaptation for object detection** [SysCV/shift-detection-tta](https://github.com/SysCV/shift-detection-tta), thanks [@Mattia Segu](https://github.com/mattiasegu)!
- **Test-time adaptation for semantic segmentation** [zwbx/SHIFT-Continuous_Test_Time_Adaptation](https://github.com/zwbx/SHIFT-Continuous_Test_Time_Adaptation/), thanks [@Wenbo Zhang](https://github.com/zwbx)!
- **3D object detection** [leaf1170124460/Mask3D-SHIFT](https://github.com/leaf1170124460/Mask3D-SHIFT), thanks [@Chengxiang Fan](https://github.com/leaf1170124460)!

## Citation

The SHIFT Dataset is made freely available to academic and non-academic entities for research purposes such as academic research, teaching, scientific publications, or personal experimentation. If you use our dataset, we kindly ask you to cite our paper as:

```
@InProceedings{shift2022,
    author    = {Sun, Tao and Segu, Mattia and Postels, Janis and Wang, Yuxuan and Van Gool, Luc and Schiele, Bernt and Tombari, Federico and Yu, Fisher},
    title     = {{SHIFT:} A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21371-21382}
}
```


Copyright © 2022, [Tao Sun](https://suniique.com) ([@suniique](https://github.com/suniique)), [ETH VIS Group](https://cv.ethz.ch/).
