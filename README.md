# example

[![Build
Status](https://travis-ci.org/tcavazos2/MachineLearn_TF.svg?branch=master)](https://travis-ci.org/tcavazos2/MachineLearn_TF)

Project to distinguish real binding sites of a yeast transcription factor Rap1 from other sequences. Input is positive transcription factor examples for yeast Rap1 and negative examples where Rap1 does not bind. 

## usage

To use the package, first make a new conda environment and activate it

```
conda create -n exampleenv python=3
source activate exampleenv
```

then run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `learn/__main__.py`) can be run as follows

```
# Run initial alignment 
python -m learn

```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.
