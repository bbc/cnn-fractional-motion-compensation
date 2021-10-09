# Interpreting super-resolution CNNs for sub-pixel motion compensation in video coding
| [Luka Murn][LukaMurn-web]  | [Saverio Blasi][SaverioBlasi-web] | [Alan F. Smeaton][AlanSmeaton-web] | [Noel E. O’Connor][NoelOConnor-web] | [Marta Mrak][MartaMrak-web] |

[LukaMurn-web]: https://www.bbc.co.uk/rd/people/luka-murn
[SaverioBlasi-web]: https://www.bbc.co.uk/rd/people/saverio-blasi
[MartaMrak-web]: https://www.bbc.co.uk/rd/people/marta-mrak
[AlanSmeaton-web]: https://www.insight-centre.org/users/alan-smeaton
[NoelOConnor-web]: https://www.insight-centre.org/our-team/prof-noel-oconnor/

![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2Fbbc%2Fcnn-fractional-motion-compensation)

## Description
The versatility of recent machine learning approaches makes them ideal for improvement of 
next generation video compression solutions. Unfortunately, these approaches typically bring significant increases 
in computational complexity and are difficult to interpret into explainable models, affecting their potential 
for implementation within practical video coding applications. This paper introduces 
a novel explainable neural network-based inter-prediction scheme, to improve the interpolation of reference samples 
needed for fractional precision motion compensation. The approach requires a single neural network to be trained 
from which a full quarter-pixel interpolation filter set is derived, as the network is easily interpretable due to 
its linear structure. A novel training framework enables each network branch to resemble a specific fractional shift. 
This practical solution makes it very efficient to use alongside conventional video coding schemes. When implemented in 
the context of the state-of-the-art Versatile Video Coding (VVC) test model, 0.77%, 1.27% and 2.25% BD-rate savings 
can be achieved on average for lower resolution sequences under the random access, low-delay B and low-delay P 
configurations, respectively, while the complexity of the learned interpolation schemes is significantly reduced 
compared to the interpolation with full CNNs.

![approach]

[approach]: readme-resources/architectures/approach.png

## Blog post and interactive demo
A summary of our work is presented in a BBC R&D blog post 
[*Interpreting Convolutional Neural Networks for Video Coding*](https://www.bbc.co.uk/rd/blog/2020-06-interpretable-ai-neural-networks-video-coding), 
as well as in an interactive demo [*Visual Data Analytics*](https://www.bbc.co.uk/taster/pilots/bbc-rd-2020-showcase), 
available on BBC Taster.

## Publications
The software in this repository presents methods from:
- ***Improved CNN-based Learning of Interpolation Filters for Low-Complexity Inter Prediction in Video Coding***,
  available on [IEEE Xplore](https://ieeexplore.ieee.org/document/9456071) and [arXiv](https://arxiv.org/abs/2106.08936);
- ***Interpreting CNN for Low Complexity Learned Sub-pixel Motion Compensation in Video Coding***,
  available on [IEEE Xplore](https://ieeexplore.ieee.org/document/9191193) and [arXiv](https://arxiv.org/abs/2006.06392);
- ***Interpreting Convolutional Neural Networks for Video Coding***,
  presented at Open Source Software Competition of ACM Multimedia 2021.

Please cite this work as:

```
@ARTICLE{Murn2021,
  author={Murn, Luka and Blasi, Saverio Giovanni and Smeaton, Alan and Mrak, Marta},
  journal={IEEE Open Journal of Signal Processing}, 
  title={Improved CNN-based Learning of Interpolation Filters for Low-Complexity Inter Prediction in Video Coding}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/OJSP.2021.3089439}}
```

## How to use

### Dependencies and Docker installation
The code is compatible with Python 3.7 and TensorFlow 1.14.0.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The repository is also available as a Docker container. Run the following commands:

```bash
cd docker
docker-compose up -d
```

Note: Docker and Docker Compose are required, and NVIDIA-Docker is preferred.

The learned interpolation filters are implemented within [VVC Test Model (VTM)](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM) 6.0 
as a switchable filter set, allowing the encoder to choose between the new filters derived from a neural network and 
the traditional filters when compressing a video sequence. To apply the patch with the necessary changes, run: 

```bash
git clone --depth 1 -b VTM6.0 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
cd VVCSoftware_VTM
git am -3 -k [path-to-this-repository]/vtm6.0-patch/cnn-fractional-motion-compensation.patch
```

Instructions on how to build platform-specific files for the VTM reference software 
are available in ```VVCSoftware_VTM/README.md```.

### How to replicate results from the research papers using pre-trained models
The coefficients of the trained neural network-based interpolation filters are already included with the applied patch.
To enable the conditions under which the switchable filter implementation is tested, 
the macro **SWITCH_IF**, located in ```VVCSoftware_VTM/source/Lib/CommonLib/TypeDef.h```, needs to be set to 1.
The learned interpolation filter set coefficients are obtained from a *CompetitionCNN* network.
Additionally, macro **PER_QP** adds multiple learned interpolation filter sets, obtained from a *ScratchCNN* network.
Both networks were trained on motion info extracted from the *BlowingBubbles* YUV video sequence.
To run the modified VTM encoder, the following command arguments need to be added:

```bash
--Triangle=0, --Affine=0, --DMVR=0, --BIO=0, --WeightedPredP=0, --WeightedPredB=0, --MHIntra=0, --SBT=0, --MMVD=0, --SMVD=0, --IMV=0, --SubPuMvp=0, --TMVPMode=0
```

The VTM coding performance of the pre-trained *CompetitionCNN* or *ScratchCNN* model needs to be compared with 
the anchor generated with the same command arguments, but with the macro **SWITCH_IF** set to 0.

### How to replicate results from the research papers starting anew 
#### Data preparation
Training data is generated from a common test video sequence in VVC, *BlowingBubbles*.
Details on how to access this YUV sequence are described in a technical report, 
[JVET-J1010: JVET common test conditions and software reference configurations](https://www.researchgate.net/publication/326506581_JVET-J1010_JVET_common_test_conditions_and_software_reference_configurations).

#### Encoding and storing data
In order to collect data necessary for creating a dataset, 
the video sequence first needs to be encoded in a modified VTM 6.0 environment.
The provided patch contains an **ENCODER_DATAEXTRACTION** macro, used to print out 
relevant fractional motion information on the encoder side of the selected video sequence.
This macro should only be run during the encoding process.
To replicate the results reported in the publications, use the *BlowingBubbles* YUV video sequence 
with the low-delay P and random access encoding configurations, and QPs 22, 27, 32, 37.
However, any video sequence can be used to generate training or testing data, 
but make sure to convert the chosen video to a YUV 420p 8bit representation. 
The encoder needs to be run with the following command arguments:

```bash
--Triangle=0, --Affine=0, --DMVR=0, --BIO=0, --WeightedPredP=0, --WeightedPredB=0
```

After saving the encoder log file generated by VTM and the decoded YUV sequence, organise the data as follows:

```bash
experiments
└── [sequence]
    └── original.yuv
    └── ld_P_main10
    │   ├── decoded_[qp].yuv
    │   └── encoder_[qp].log
    └── ra_main10
        ├── decoded_[qp].yuv
        └── encoder_[qp].log
```
A template directory structure is provided in [experiments](./experiments).

#### Creating datasets
To create a dataset from the coded video sequence, add its details to a file in [dataset-configs](./dataset-configs), 
and define the dataset path.
Details are already provided for the *BlowingBubbles* YUV sequence, and 
a template file is available for an arbitrary sequence.
Following this, run commands:

```bash
cd dataset-creation
python dataset_fractional_me.py -c ../dataset-configs/[sequence-details].py
```

#### Training and testing models
A number of different models are available for training purposes, as explained in the publications.
Their architectures generally correspond to:

| ![(a) ScratchCNN / SRCNN][scratch-arch] | ![(b) SharedCNN][shared-arch] | ![(c) CompetitionCNN][competition-arch]
|:-:|:-:|:-:|
| (a) ScratchCNN / SRCNN  | (b) SharedCNN | (c) CompetitionCNN

[scratch-arch]: readme-resources/architectures/scratch-arch.png
[shared-arch]: readme-resources/architectures/shared-arch.png
[competition-arch]: readme-resources/architectures/competition-arch.png

The hyperparameters and the dataset used for training / testing these architectures 
are defined in [model-configs](./model-configs), with a template available for any arbitrary models. 
Once the datasets are created, train or test the desired model by running the following command:

```bash
python main.py -c model-configs/[model-name].py -a [train/test]
```

#### Applying the learned filters
In order to integrate the learned interpolation filters into VTM 6.0, their coefficients need to be extracted 
from the trained models, as visualised:

![filter-extraction]

After training a certain architecture on a dataset, run the commands:

```bash
cd tools
python load_learned_filters.py -m [model]
```

An array of filter coefficients will be stored in the results directory, as defined in the model configuration file.
Copy the array to ```VVCSoftware_VTM/source/Lib/CommonLib/InterpolationFilter.cpp``` in the patched VTM 6.0 codec 
and run the encoder with the same command arguments as specified in 
[Pre-trained models](#how-to-replicate-results-from-the-research-papers-using-pre-trained-models).
An anchor needs to be generated with the same command arguments and encoding restrictions.


[filter-extraction]: readme-resources/filter-extraction.png

#### Analyzing the usage of the learned filters
After implementing the learned filters in the modified VTM 6.0 codec, a statistics collector macro
**DECODER_STATISTICS** can be enabled, to print out relevant details on the usage of said filters 
for a particular video sequence. This macro should only be run during the decoding process. 
The saved log file and the decoded video sequence should be arranged as follows:

```bash
experiments
└── [sequence]
    └── original.yuv
    └── encoding-configuration
        ├── decoded-switchable_[qp].yuv
        └── decoder-switchable_[qp].log
```

To analyze the collected statistics, 
define the details in [tools-configs/decoder_stats.py](./tools-configs/decoder_stats.py) 
and run the following commands:

```bash
cd tools
python analyze_decoder_statistics.py -c ../tools-configs/decoder_stats.py
```

This script will create a file with the hit ratio of the learned filters for a particular video sequence, 
alongside a visualisation of the filter usage for each frame of the decoded YUV video, as seen in this example:

![filter-usage]

[filter-usage]: readme-resources/filter-usage.png

## Acknowledgements
This work has been conducted within the project JOLT. This project is funded by the European Union’s Horizon 2020 
research and innovation programme under the Marie Skłodowska Curie grant agreement No 765140.

| ![JOLT-photo] | ![EU-photo] |
|:-:|:-:|
| [JOLT Project][JOLT-web] | [European Commission][EU-web] |


[JOLT-photo]: readme-resources/logos/jolt.png "JOLT"
[EU-photo]: readme-resources/logos/eu.png "European Commission"


[JOLT-web]: http://joltetn.eu/
[EU-web]: https://ec.europa.eu/programmes/horizon2020/en

A collaboration between:

| ![logo-bbc] | ![logo-dcu] | ![logo-insight] |
|:-:|:-:|:-:|
| [BBC Research & Development][bbc-web] | [Dublin City University (DCU)][dcu-web] | [Insight Centre for Data Analytics][insight-web] |

[bbc-web]: https://www.bbc.co.uk/rd
[insight-web]: https://www.insight-centre.org/
[dcu-web]: http://www.dcu.ie/

[logo-bbc]: readme-resources/logos/bbcrd.png  "BBC Research & Development"
[logo-insight]: readme-resources/logos/insight.png "Insight Centre for Data Analytics"
[logo-dcu]: readme-resources/logos/dcu.png "Dublin City University"

## Contact
If you have any general questions about our work or code which may be of interest to other researchers, 
please use the [public issues section](https://github.com/bbc/cnn-fractional-motion-compensation/issues) 
of this GitHub repository. Alternatively, e-mail us at <mailto:luka.murn@bbc.co.uk>.
