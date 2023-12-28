import matplotlib.pyplot as plt
from textwrap import wrap
from transformers import AutoProcessor
from evaluate import load
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn

device='cuda' if torch.cuda.is_available() else 'cpu'
wer = load("wer")

def plot_images(images, captions):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        caption = captions[i]
        caption = "\n".join(wrap(caption, 12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis("off")

def transforms(example_batch,processor):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs

def compute_metrics(eval_pred,processor):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}

class ImageCaptioningDataset2(Dataset):
  def __init__(self,dataset,processor):
    self.dataset=dataset
    self.processor=processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self,idx):
    item=self.dataset[idx]
    encoding1=self.processor(images=item['image_0'],text=item['caption_0'],padding='max_length',return_tensors='pt')
    encoding2=self.processor(images=item['image_1'],text=item['caption_1'],padding='max_length',return_tensors='pt')
    #remove batch dimension
    encoding1={k:v.squeeze() for k,v in encoding1.items()}
    encoding2={k:v.squeeze() for k,v in encoding2.items()}
    return [encoding1,encoding2]

def train_2(num_epochs,model,optimizer,train_dataloader_1,train_dataloader_2,val_dataloader_1,val_dataloader_2):

  epoch_train_loss_lst=[]
  epoch_val_loss_lst=[]
  for epoch in range(num_epochs):
    print("Epoch:", epoch)
    batch_train_loss=0
    batch_val_loss=0
    for idx, batch in enumerate(train_dataloader_1):
      input_ids = batch.pop("input_ids").to(device)
      pixel_values = batch.pop("pixel_values").to(device)

      outputs = model(input_ids=input_ids,
                      pixel_values=pixel_values,
                      labels=input_ids)

      loss = outputs.loss
      batch_train_loss+=loss.item()
      print("Loss:", loss.item())

      loss.backward()

      optimizer.step()
      optimizer.zero_grad()
    for idx, batch in enumerate(train_dataloader_2):
      input_ids = batch.pop("input_ids").to(device)
      pixel_values = batch.pop("pixel_values").to(device)

      outputs = model(input_ids=input_ids,
                      pixel_values=pixel_values,
                      labels=input_ids)

      loss = outputs.loss
      batch_train_loss+=loss.item()
      print("Loss:", loss.item())

      loss.backward()

      optimizer.step()
      optimizer.zero_grad()
    epoch_train_loss_lst.append(batch_train_loss)
    for idx,batch in enumerate(val_dataloader_1):
      input_ids = batch.pop("input_ids").to(device)
      pixel_values = batch.pop("pixel_values").to(device)
      outputs_val=model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids
          )
      val_loss=outputs_val.loss
      batch_val_loss+=val_loss.item()
      print('val loss',val_loss)
    for idx,batch in enumerate(val_dataloader_2):
      input_ids = batch.pop("input_ids").to(device)
      pixel_values = batch.pop("pixel_values").to(device)
      outputs_val=model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids
          )
      val_loss=outputs_val.loss
      batch_val_loss+=val_loss.item()
      print('val loss',val_loss)
    epoch_val_loss_lst.append(batch_val_loss)

  return epoch_train_loss_lst, epoch_val_loss_lst,model
  


