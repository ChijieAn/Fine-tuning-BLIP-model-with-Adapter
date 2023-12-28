from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from evaluate import load
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch.utils.data import Dataset,DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration,BlipConfig
import torch
import copy
import os
import matplotlib.pyplot as plt

from functions import (plot_images,transforms,evaluate,ImageCaptioningDataset2,train_2)
from model import (MyAdapter,ConcatenatedModel)

device='cuda' if torch.cuda.is_available() else 'cpu'

#load the dataset (this is only a sample to display some images in the pokemon dataset)
ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

#plot some sample images in the dataset
sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
sample_captions = [train_ds[i]["text"] for i in range(5)]
plot_images(sample_images_to_visualize, sample_captions)

#import the processor
checkpoint = "microsoft/git-base"
processor = AutoProcessor.from_pretrained(checkpoint)

#transform the images
train_ds.set_transform(transforms)
test_ds.set_transform(transforms)

wer = load("wer")

#load the facebook image caption dataset for training
dataset2 = load_dataset('facebook/winoground', use_auth_token='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',split='test')

#split the training and validation dataset
train_val_dataset2,test_dataset2=train_test_split(dataset2,test_size=0.1,random_state=42)

#create the training an validation dataset
train_val_dataset2=Dataset.from_dict(train_val_dataset2)
test_dataset2=Dataset.from_dict(test_dataset2)

train_dataset2, val_dataset2 = train_test_split(train_val_dataset2, test_size=0.1, random_state=42)
train_dataset2=Dataset.from_dict(train_dataset2)
val_dataset2=Dataset.from_dict(val_dataset2)

#get the  caption dataset, and split it into train, validation and test
encoding_train=ImageCaptioningDataset2(train_dataset2,processor)
encoding1_train=encoding_train[0]
encoding2_train=encoding_train[1]
train_dataloader_1 = DataLoader(encoding1_train, shuffle=True, batch_size=2)
train_dataloader_2 = DataLoader(encoding2_train, shuffle=True, batch_size=2)
encoding_val=ImageCaptioningDataset2(val_dataset2,processor)
encoding1_val=encoding_val[0]
encoding2_val=encoding_val[1]
val_dataloader_1 = DataLoader(encoding1_val, shuffle=True, batch_size=2)
val_dataloader_2 = DataLoader(encoding2_val, shuffle=True, batch_size=2)
encoding_test=ImageCaptioningDataset2(test_dataset2,processor)
encoding1_test=encoding_test[0]
encoding2_test=encoding_test[1]
test_dataloader_1 = DataLoader(encoding1_test, shuffle=True, batch_size=2)
test_dataloader_2 = DataLoader(encoding2_test, shuffle=True, batch_size=2)

#load the model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_base = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
config=BlipConfig.from_pretrained("Salesforce/blip-image-captioning-base")

#add the adapter layer in pretrained encoder layer 0-11
adapter_0=MyAdapter(768,512,768)
new_blipMLP_0=ConcatenatedModel(model_base.vision_model.encoder.layers[0].mlp,adapter_0)
model_base.vision_model.encoder.layers[0].mlp=new_blipMLP_0

adapter_1=MyAdapter(768,512,768)
new_blipMLP_1=ConcatenatedModel(model_base.vision_model.encoder.layers[1].mlp,adapter_1)
model_base.vision_model.encoder.layers[1].mlp=new_blipMLP_1

adapter_2=MyAdapter(768,512,768)
new_blipMLP_2=ConcatenatedModel(model_base.vision_model.encoder.layers[2].mlp,adapter_2)
model_base.vision_model.encoder.layers[2].mlp=new_blipMLP_2

adapter_3=MyAdapter(768,512,768)
new_blipMLP_3=ConcatenatedModel(model_base.vision_model.encoder.layers[3].mlp,adapter_3)
model_base.vision_model.encoder.layers[3].mlp=new_blipMLP_3

adapter_4=MyAdapter(768,512,768)
new_blipMLP_4=ConcatenatedModel(model_base.vision_model.encoder.layers[4].mlp,adapter_4)
model_base.vision_model.encoder.layers[4].mlp=new_blipMLP_4

adapter_5=MyAdapter(768,512,768)
new_blipMLP_5=ConcatenatedModel(model_base.vision_model.encoder.layers[5].mlp,adapter_5)
model_base.vision_model.encoder.layers[5].mlp=new_blipMLP_5

adapter_6=MyAdapter(768,512,768)
new_blipMLP_6=ConcatenatedModel(model_base.vision_model.encoder.layers[6].mlp,adapter_6)
model_base.vision_model.encoder.layers[6].mlp=new_blipMLP_6

adapter_7=MyAdapter(768,512,768)
new_blipMLP_7=ConcatenatedModel(model_base.vision_model.encoder.layers[7].mlp,adapter_7)
model_base.vision_model.encoder.layers[7].mlp=new_blipMLP_7

adapter_8=MyAdapter(768,512,768)
new_blipMLP_8=ConcatenatedModel(model_base.vision_model.encoder.layers[8].mlp,adapter_8)
model_base.vision_model.encoder.layers[8].mlp=new_blipMLP_8

adapter_9=MyAdapter(768,512,768)
new_blipMLP_9=ConcatenatedModel(model_base.vision_model.encoder.layers[9].mlp,adapter_9)
model_base.vision_model.encoder.layers[9].mlp=new_blipMLP_9

adapter_10=MyAdapter(768,512,768)
new_blipMLP_10=ConcatenatedModel(model_base.vision_model.encoder.layers[10].mlp,adapter_10)
model_base.vision_model.encoder.layers[10].mlp=new_blipMLP_10

adapter_11=MyAdapter(768,512,768)
new_blipMLP_11=ConcatenatedModel(model_base.vision_model.encoder.layers[11].mlp,adapter_11)
model_base.vision_model.encoder.layers[11].mlp=new_blipMLP_11

#lock the parameters in the parts other from the adapter in the blip vision model(image encoder)
for i in range(12):
  for param in model_base.vision_model.encoder.layers[i].self_attn.parameters():
    param.requires_grad=False
  for param in  model_base.vision_model.encoder.layers[i].mlp.net1.parameters():
    param.requires_grad=False

#Also add adapters to the dense layer in the original model(11 in total)
adapter_1_0=MyAdapter(768,512,768)
new_blipDense_0=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[0].output.dense,adapter_1_0)
model_base.text_decoder.bert.encoder.layer[0].output.dense=new_blipDense_0

adapter_1_1=MyAdapter(768,512,768)
new_blipDense_1=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[1].output.dense,adapter_1_1)
model_base.text_decoder.bert.encoder.layer[1].output.dense=new_blipDense_1

adapter_1_2=MyAdapter(768,512,768)
new_blipDense_2=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[2].output.dense,adapter_1_2)
model_base.text_decoder.bert.encoder.layer[2].output.dense=new_blipDense_2

adapter_1_3=MyAdapter(768,512,768)
new_blipDense_3=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[3].output.dense,adapter_1_3)
model_base.text_decoder.bert.encoder.layer[3].output.dense=new_blipDense_3

adapter_1_4=MyAdapter(768,512,768)
new_blipDense_4=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[4].output.dense,adapter_1_4)
model_base.text_decoder.bert.encoder.layer[4].output.dense=new_blipDense_4

adapter_1_5=MyAdapter(768,512,768)
new_blipDense_5=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[5].output.dense,adapter_1_5)
model_base.text_decoder.bert.encoder.layer[5].output.dense=new_blipDense_5

adapter_1_6=MyAdapter(768,512,768)
new_blipDense_6=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[6].output.dense,adapter_1_6)
model_base.text_decoder.bert.encoder.layer[6].output.dense=new_blipDense_6

adapter_1_7=MyAdapter(768,512,768)
new_blipDense_7=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[7].output.dense,adapter_1_7)
model_base.text_decoder.bert.encoder.layer[7].output.dense=new_blipDense_7

adapter_1_8=MyAdapter(768,512,768)
new_blipDense_8=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[8].output.dense,adapter_1_8)
model_base.text_decoder.bert.encoder.layer[8].output.dense=new_blipDense_8

adapter_1_9=MyAdapter(768,512,768)
new_blipDense_9=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[9].output.dense,adapter_1_9)
model_base.text_decoder.bert.encoder.layer[9].output.dense=new_blipDense_9

adapter_1_10=MyAdapter(768,512,768)
new_blipDense_10=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[10].output.dense,adapter_1_10)
model_base.text_decoder.bert.encoder.layer[10].output.dense=new_blipDense_10

adapter_1_11=MyAdapter(768,512,768)
new_blipDense_11=ConcatenatedModel(model_base.text_decoder.bert.encoder.layer[11].output.dense,adapter_1_11)
model_base.text_decoder.bert.encoder.layer[11].output.dense=new_blipDense_11

#freze the parameters in the parts other than the self-defined parameter in the model
for i in range(12):
  for params in model_base.text_decoder.bert.encoder.layer[i].output.dense.net1.parameters():
    params.requires_grad=False
  for params in model_base.text_decoder.bert.encoder.layer[i].intermediate.parameters():
    params.requires_grad=False
  for params in model_base.text_decoder.bert.encoder.layer[i].crossattention.parameters():
    params.requires_grad=False
  for params in model_base.text_decoder.bert.encoder.layer[i].attention.parameters():
    params.requires_grad=False

for params in model_base.text_decoder.bert.embeddings.parameters():
  params.requires_grad=False

for params in model_base.text_decoder.cls.parameters():
  params.requires_grad=False

#save the parameters of the base model
params_dict = model_base.state_dict()

#copy the refined model and load parameters
model_5=copy.deepcopy(model_base)
model_5.load_state_dict(params_dict)

#only train the parameters that are not freezed (the parameters in the adapter)
optimizer_5=torch.optim.AdamW(filter(lambda p: p.requires_grad, model_5.parameters()),lr=5e-5)

model_5.to(device)
epoch_train_loss_lst,epoch_val_loss_lst,model_5=train_2(num_epochs=5,model=model_5,optimizer=optimizer_5,train_dataloader_1=train_dataloader_1,train_dataloader_2=train_dataloader_2,val_dataloader_1=val_dataloader_1,val_dataloader_2=val_dataloader_2)

#save the model on google drive
save_dir='/content/drive/MyDrive/Machine Learning final project/models'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
torch.save(model_5,save_dir+'/model.pth')

#get the test loss
test_loss=0
#freeze the parameters
model_5.eval()
for idx, batch in enumerate(test_dataloader_1):
      input_ids = batch.pop("input_ids").to(device)
      pixel_values = batch.pop("pixel_values").to(device)

      outputs = model_5(input_ids=input_ids,
                      pixel_values=pixel_values,
                      labels=input_ids)

      loss = outputs.loss

      test_loss+=loss

for idx, batch in enumerate(test_dataloader_2):
      input_ids = batch.pop("input_ids").to(device)
      pixel_values = batch.pop("pixel_values").to(device)

      outputs = model_5(input_ids=input_ids,
                      pixel_values=pixel_values,
                      labels=input_ids)

      loss = outputs.loss

      test_loss+=loss

print(test_loss.item())

#display the results
fig = plt.figure(figsize=(18, 14))
#prepare image for the model
for i,example in enumerate(test_dataset2):
  image = example["image_0"]
  inputs = processor(images=image, return_tensors="pt").to(device)
  pixel_values = inputs.pixel_values

  generated_ids = model_5.generate(pixel_values=pixel_values, max_length=50)
  generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  print(generated_caption)



