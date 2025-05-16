## Bi-DiffCD
The implementation of the paper "Bidirectional Diffusion Guided Collaborative Change Detection for Arbitrary-Modal Remote Sensing Images"

## Prerequisites

- Ubuntu 20.04 cuda 12.2
- Python 3.7 Pytorch 2.4.0 

## Usage

Bi-DiffCD is divided into two stages: modal transformation and change detection.

	Stages1: Modal Transformation
	
		train：--run 'train_stage1.py'
	
		test：--run 'test_stage1.py'

	Stages2: Change Detection
	
		train：--run 'train_stage2.py'
	
		test：--run 'test_stage2.py'
 
## Cite
If you find this code helpful, please kindly cite:
