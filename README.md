# Fine-tuning-BLIP-model-with-Adapter
We used a method called Adapters to revised a pre trained BLIP image caption model, and fine tuned the model on a specific dataset. 

The most important file in this repository is BLIP_model_fintune_sample.ipynb, where we added some adapter layers to a pretrained BLIP model. Then we freeze the pretrained encoder and decoder for the model and fintuned the model on a specific dataset for image caption tasks. 

# The Notebooks folder
In this folder, I mainly includes some notebooks I created while doing the project.

# The Python_files folder
In this folder, there're three python files.

In functions.py, I defined some functions for training the model and doing some other tasks like plotting and processing images

In model.py, I defined the classes of an Adapter model and a cancatenated model

In main.py, I did implemented the functions and classes defined in the other two python files and fintuned a BLIP model with adapters on a Facebook image caption dataset. 
