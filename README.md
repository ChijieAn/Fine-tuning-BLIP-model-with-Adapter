# Fine-tuning-BLIP-model-with-Adapter
We used a method called Adapters to revised a pre trained BLIP image caption model, and fine tuned the model on a specific dataset. 

The most important file in this repository is BLIP_model_fintune_sample.ipynb, where we added some adapter layers to a pretrained BLIP model. Then we freeze the pretrained encoder and decoder for the model and fintuned the model on a specific dataset for image caption tasks. 
