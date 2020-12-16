#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json


# In[2]:

import os
import json
import torch
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import matplotlib.pyplot as plt
import metrics


# In[4]:


tokenizer_string = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(tokenizer_string)


# In[5]:

max_length = 80

'''
Abstracts DocVQA
'''


class DocVQA(Dataset):

    def __init__(self,
                 mode: str,
                 transform: object = None,
                 seq_len: int = max_length):
        '''
        mode: one of train, val and test.
        transform: transforms to be applied to the document image if applicable.
        seq_len: maximum sequence len of encoded tokens.

        returns:
            dict:
                document: transformed document image.
                input_tokens: tokenized text contained in the document.
                input_text: text contained in the document.
                bboxes: bounding boxes for each OCR detection in the document, on the format [tl_col, tl_row, br_col, br_row].
        '''
        super().__init__()
        assert mode in ["train", "val", "test"]
        with open(f"/docvqa/{mode}/{mode}_v1.0.json", 'r') as data_json_file:
            self.data_json = json.load(data_json_file)

        self.folder = f"/docvqa/{mode}"
        self.transform = transform
        self.seq_len = seq_len
        self.mode = mode

        print(f"{self.mode} DocVQA folder {self.folder} tokenizer {tokenizer.__class__.__name__} transform {self.transform} seq_len {self.seq_len}")

    def __len__(self):
        return len(self.data_json["data"])

    def __getitem__(self, i: int):
        data = self.data_json["data"][i]
        
        document = Image.open(os.path.join(self.folder, data["image"])).convert("RGB")
        
        question_text = data['question']
        answer_text = np.random.choice(data.get('answers', ["N/A"]))
        
        target_text = "question: " + question_text.strip() + " answer: " + answer_text.strip()
        

#         target_ids = self.tokenizer.encode(target_text,
#                                        padding='max_length',
#                                        truncation=True,
#                                        max_length=self.seq_len,
#                                        return_tensors='pt')[0]

        if self.transform is not None:
            document = self.transform(document)

        return {"document": document,
                "question": question_text,
                "answer": answer_text,
                "target_text": target_text}


# In[6]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AdamW
from torchvision import transforms
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback



class CaptioningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # EfficientNet image encoder.
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')

        self.decoder = T5ForConditionalGeneration.from_pretrained(tokenizer_string)
        # for par in self.decoder.base_model.parameters():
        #     par.requires_grad = False
        
        # Bridge convolution layer between efficientnet and transformer formats.
        self.bridge = nn.Conv2d(in_channels=112, out_channels=self.decoder.config.d_model, kernel_size=1)
        
    def _embeds_forward(self, img):

        # Retrieve the features from the last layer.
        # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        #
        # >>> img = torch.ones((1, 3, 256, 256))
        # >>> encoder.extract_features(img).shape
        # torch.Size([1, 1280, 8, 8])
        # features = self.encoder.extract_features(img)
        features = self.encoder.extract_endpoints(img)["reduction_4"] 

        # print(features.shape)

        # Compute the bridge convolution, it should reformat the inputs to feed the transformer network.
        features = self.bridge(features)
        
        # Reshape the output to match the embedding dimension of the encoder with 64 tokens.
        inputs_embeds = features             .permute(0, 2, 3, 1)             .reshape(features.shape[0], -1, self.decoder.config.d_model)

        return inputs_embeds

    def forward(self, img=None, inputs_embeds=None, decoder_input_ids=None, labels=None):

        # Pass efficientnet hidden states as embeddings for the transformer encoder input.
        inputs_embeds = self._embeds_forward(img) if inputs_embeds is None else inputs_embeds

        return self.decoder(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids, 
            labels=labels,
        )

    def generate(self, img, max_len=max_length):

        # We need to implement our own generate loop as transformers doesn't accept 
        # precomputed embeddings on the generate method.
        # Issue: https://github.com/huggingface/transformers/issues/7626
        # Precompute embeddings to speedup generation as they don't change.
        inputs_embeds = self._embeds_forward(img)
        
        decoder_input_ids = torch.full(
            (1, 1), self.decoder.config.decoder_start_token_id, dtype=torch.long, device=img.device
        )
        
        for i in range(max_len):
            with torch.no_grad():
                output = self.forward(decoder_input_ids=decoder_input_ids, 
                                      inputs_embeds=inputs_embeds)

                logits = output[0]
                next_token_logits = logits[:, -1, :]
                next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1).to(img.device)

                if torch.eq(next_token_id[:, -1], self.decoder.config.eos_token_id).all():
                    break

        return decoder_input_ids

    def training_step(self, batch, batch_idx):
        img, targets, raw_text = batch
        output = self(img, labels=targets)
        return output[0]

    def validation_step(self, batch, batch_idx):
        img, targets, raw_text = batch
        
        tokens = [self.generate(im.view((1,) + im.shape))[0].cpu() for im in img]

        with torch.no_grad():        
            loss_val = self(img, labels=targets)[0].item()

        return (tokens, raw_text, img, loss_val)

    def validation_epoch_end(self, validation_step_outputs):
        
        validation_step_outputs = list(validation_step_outputs)
    
        tokens_batch = [t for out in validation_step_outputs for t in out[0]]
        reference_batch = [r for out in validation_step_outputs for r in out[1]]
        # img_batch = [i for out in validation_step_outputs for i in out[2]]
        hist_loss_val = np.mean([out[3] for out in validation_step_outputs])
        
        generated_batch = tokenizer.batch_decode(tokens_batch)
        

        F1_val = np.mean([metrics.compute_f1(gold, pred) for gold, pred in zip(reference_batch,
                                                                    generated_batch)])


        exact_val = np.mean([metrics.compute_exact(gold, pred) for gold, pred in zip(reference_batch,
                                                                            generated_batch)])
        
        self.log("loss_val", torch.Tensor([hist_loss_val]).to(self.device), prog_bar=True,sync_dist=True)
        self.log("exact_val", torch.Tensor([exact_val]).to(self.device), prog_bar=True,sync_dist=True)
        self.log("F1_val", torch.Tensor([F1_val]).to(self.device), prog_bar=True,sync_dist=True)

    def test_step(self, batch, batch_idx):
        img, targets, raw_text = batch
        
        tokens = [self.generate(im.view((1,) + im.shape))[0].cpu() for im in img]
        
        return (tokens, raw_text, img.cpu())
    
    def test_epoch_end(self, test_step_outputs):
        test_step_outputs = list(test_step_outputs)
    
        tokens_batch = [t for out in test_step_outputs for t in out[0]]
        reference_batch = [r for out in test_step_outputs for r in out[1]]
        
        generated_batch = tokenizer.batch_decode(tokens_batch)
        
        F1_test = np.mean([metrics.compute_f1(gold, pred) for gold, pred in zip(reference_batch,
                                                                    generated_batch)])


        exact_test = np.mean([metrics.compute_exact(gold, pred) for gold, pred in zip(reference_batch,
                                                                            generated_batch)])
        
        self.log("exact_test", torch.Tensor([exact_test]).to(self.device), prog_bar=True,sync_dist=True)
        self.log("F1_test", torch.Tensor([F1_test]).to(self.device), prog_bar=True,sync_dist=True)

        return {"exact_test": exact_test,
                "F1_test": F1_test}
    
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=5e-4)


# # Experimentando, testando e salvando

# In[7]:


def collate_fn(batch):
    """
    Input any (selected randomly) caption sample for every image. Useful for training.
    """
    #lista de [imagem, token_ids, texto_original]
    imgs = [r['document'].numpy() for r in batch]
    texts = [r['target_text'] for r in batch]


    tokens_ids = tokenizer.batch_encode_plus(texts,
                                               truncation=True, 
                                               return_tensors="pt", 
                                               padding="max_length",
                                               max_length=max_length)['input_ids']
    
    return (
        torch.Tensor(imgs),
        tokens_ids, 
        texts,
    )


val_test_transforms = transforms.Compose([transforms.Resize((700, 400)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])
                        ])


#cria os datasets necessáios
dataset_train = DocVQA(mode='train', transform=val_test_transforms)

# In[8]:



# In[9]:



# In[ ]:

train_transforms = transforms.Compose([transforms.Resize((700, 400)),
                                    transforms.ColorJitter(0.2, 0.3, 0.3, 0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])
                        ])


val_test_transforms = transforms.Compose([transforms.Resize((700, 400)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])
                        ])


#cria os datasets necessáios
dataset_train = DocVQA(mode='train', transform=train_transforms)
dataset_val = DocVQA(mode='val', transform=val_test_transforms)
dataset_test = DocVQA(mode='test', transform=val_test_transforms)

batch = 2

num_workers = 8
dataloader_train = DataLoader(dataset_train, 
                                batch_size=batch,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers,
                                collate_fn=collate_fn)

dataloader_val = DataLoader(dataset_val, 
                            batch_size=batch,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)


dataloader_test = DataLoader(dataset_test, 
                            batch_size=batch,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)


# path_to_checkpoint= '/captioning-epoch=14-loss_val=0.68-docvqa80.ckpt'
path_to_checkpoint = "/logs/"

if not os.path.isfile(path_to_checkpoint):
    path_to_checkpoint = None
else:
    print(f"Logging from: {path_to_checkpoint}")

# Log results to CSV (so we can plot them later).
logger = pl.loggers.csv_logs.CSVLogger(f"./logs", name="vqa-generation")


# Checkpoint the best model.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    prefix="captioning",
    filepath="logs/{epoch}-{F1_val:.2f}-docvqa"+str(max_length), 
    monitor="F1_val", 
    mode="min"
)


model = CaptioningModule()
model_trainer = pl.Trainer(gpus=2, 
                           max_epochs=5,
                           callbacks=[checkpoint_callback],
                           logger=logger,
                           resume_from_checkpoint=path_to_checkpoint,
                           accumulate_grad_batches=2,
                           check_val_every_n_epoch=1,
                           limit_val_batches=0.10,
                          accelerator='ddp')
model_trainer.fit(model, dataloader_train, dataloader_val)

print("Test Stage....")

model_trainer.test(model, dataloader_test)

