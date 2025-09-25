# Fine-Tune SAM2.1 for Segmentation

SAM2 is a state-of-the-art segmentation model developed by Meta. Link: 
We can fine-tune a pre-trained model on our custom data. We have provided a **notebook** that goes through the process step-by-step.

## Installation:

Clone the git repository in your local space:

Install SAM2:

Download the pre-trained checkpoint:


## Data:

- Extract the relevant data that contains the objects the model needs to learn and be finetuned on:

The data structure format required for training is of the form:
- train
	- image1.jpg
	- image1.json
	...
- val
	- image100.jpg
	- image100.json

Where the json files contain the corresponding image filenames and segmentation run-length-encodings.

- Change the paths in the data.yaml file to these respectively.

## Train:

We can now run training by running the command:

Note: The eperiments were performed on an A100 GPU and 2 L40s GPUs on a high performance cluster.


## Inference

We can view both qualitative and quantitative results on our fine-tuned model by running inference with them.

- For qualitative results, we generate segmentation masks using SAM2's Automaskgenerator library.

- For quantitative results, we can test it on our provided test-set using an evaluation script:

 
