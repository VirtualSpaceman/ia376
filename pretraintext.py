#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import multiprocessing as mp
from sys import argv

import torch
from torch import nn
import numpy as np

import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
from transformers import AdamW


from Datasets import WikiTable, WikiTableText, SquadDataset, Modes
from metrics import compute_exact, compute_f1


# In[9]:


class CNNTransformer(pl.LightningModule):
    """
        Class for combinining Vision and Language embeddings for generating answers in VQA Task
    """
    def __init__(self, hparams):
        super().__init__()
        
        #Parameters stored in dictionary
        self.hparams = hparams
        
        #Tokenizer for decoding sentences
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.t5_model)
        
        #Decoder -> Decode image embedding combined with the last hidden state of the encoder
        self.decoder = T5ForConditionalGeneration.from_pretrained(self.hparams.t5_model)
        
        #Sentence encoder -> just transformer encoder for questions
        if self.hparams.same_enc:
            self.sentence_encoder = self.decoder.get_encoder()
        else:
            self.sentence_encoder = T5EncoderModel.from_pretrained(self.hparams.t5_model)
        
        self.sync_dist = self.hparams.gpus > 1
    
    def forward(self, batch):
        input_ids = batch['input_ids']
        input_attn_mask = batch['input_attn_mask']
        questions = batch['question']
        answers = batch['answer']
        target_ids = batch['target_ids']
        
        
        #obtain the sentence encoder outputs
        encoder_outputs = self.sentence_encoder(input_ids=input_ids,
                                                attention_mask=input_attn_mask,
                                                output_attentions=self.hparams.use_enc_attn)
        
        #batch size x seqlen x self.d_model
        encoder_hidden_state = encoder_outputs.last_hidden_state
        
        #perhaps use attention coming from the sentence encoder
        encoder_attn = encoder_outputs.attentions if self.hparams.use_enc_attn else None 
    
        if self.training:
            loss = self.decoder(encoder_outputs=(encoder_hidden_state, encoder_attn),
                                labels=target_ids).loss
            
            return loss
        else:
            
            return self.generate_predictions(hiddn_states=encoder_hidden_state,
                                             encoder_attentions=encoder_attn)
        
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        
        self.log('loss', torch.Tensor([loss]).to(self.device), on_epoch=True, 
                 on_step=True, prog_bar=True, sync_dist=self.sync_dist)
        return loss
    
    
    def generate_predictions(self, hiddn_states, encoder_attentions=None):
        '''
        Adaptação de código da aula 10 do semestre passado.
        Usa features construídas externamente para gerar frases com T5.
        '''
        #max len for generated sequence
        max_seq_len = self.hparams.max_decod_len
        
        #decoded ids. Initial tokens for decoding for each batch
        decoded_ids = torch.full((hiddn_states.size(0), 1),
                                 self.decoder.config.decoder_start_token_id,
                                 dtype=torch.long).to(hiddn_states.device)
        
        #combined hidden states (image + text)
        encoder_hidden_states = hiddn_states
        
        #decoding time!
        for step in range(max_seq_len):
            #get the next token id given the ones decoded so far
            outputs = self.decoder(decoder_input_ids=decoded_ids,
                                  encoder_outputs=(encoder_hidden_states, encoder_attentions),
                                  return_dict=True)
            #take the logits
            logits = outputs["logits"]
            
            #get last logits
            next_token_logits = logits[:, -1, :]

            # Greedy decoding
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)

            # Check if output is end of senquence for all batches
            if torch.eq(next_token_id[:, -1], self.tokenizer.eos_token_id).all():
                break

            # Concatenate past ids with new id, keeping batch dimension
            decoded_ids = torch.cat([decoded_ids, next_token_id], dim=-1)

        return decoded_ids
    
    
    def evaluation_step(self, batch):
        '''
        Same step for validation and testing.
        '''
        #get the predictions
        pred_tokens = self(batch)
        preds = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True) 

        return batch["answer"], preds
    
    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch)

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch)
    
    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self.epoch_end(outputs, "test")
    
    def epoch_end(self, outputs, step):
        tgts, preds = [], []
        for output in outputs:
            tgts += output[0]
            preds += output[1]

        f1s, exacts = [], []
        for tgt, pred in zip(tgts, preds):
            f1s.append(compute_f1(tgt, pred))
            exacts.append(compute_exact(tgt, pred))
        if self.hparams.debug:
            print(f"Preds: {preds}")
            print(f"Real: {tgts}")
        else:
            self.logger.experiment.log_text("Outputs", f"Preds: {preds[-10:]} \n Real: {tgts[-10:]} \n\n")

        
        self.log(f"f1_{step}", torch.Tensor([np.array(f1s).mean()]).to(self.device),
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        
        self.log(f"{step}_exact_match", torch.Tensor([np.array(exacts).mean()]).to(self.device),
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
         
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        ds = None
        if self.hparams.pretrain:
            print(f"Pretrainig using {self.hparams.squad} - Train")
            ds = SquadDataset(Modes.TRAIN, self.tokenizer, self.hparams.squad, max_len=self.hparams.seq_len)
        else:
            ds = WikiTableText(Modes.TRAIN, self.tokenizer, max_len=self.hparams.seq_len)
            print(f"Training using {ds.__class__.__name__}")
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.nworkers)

    def val_dataloader(self):
        ds = None
        if self.hparams.pretrain:
            ds = SquadDataset(Modes.VAL, self.tokenizer, self.hparams.squad, max_len=self.hparams.seq_len)
            print(f"Pretrainig using {self.hparams.squad} - Valid")
        else:
            ds = WikiTableText(Modes.VAL, self.tokenizer, max_len=self.hparams.seq_len)
            print(f"Validating with {ds.__class__.__name__}")
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.nworkers)

    def test_dataloader(self):
        ds = None
        if self.hparams.pretrain:
            print(f"Pretrainig using {self.hparams.squad} - Test")
            ds = SquadDataset(Modes.TEST, self.tokenizer, self.hparams.squad, max_len=self.hparams.seq_len)
        else:
            ds = WikiTableText(Modes.TEST, self.tokenizer, max_len=self.hparams.seq_len)
            print(f"Testing with {ds.__class__.__name__}")
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.nworkers)


# In[10]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action='store_true', help="Pretrain or Train")
    parser.add_argument("--t5_model", type=str, default="t5-base", help="T5 weights to load.")
    parser.add_argument("--seq_len", type=int, default=128, help="Transformer sequence length.")
    parser.add_argument("--max_decod_len", type=int, default=32, help="Fast dev run mode.")
    parser.add_argument("--lr", type=float, default=5e-4, help="ADAM Learning Rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs.")
    parser.add_argument("--acum", type=int, default=1, help="Acum for batch.")
    parser.add_argument("--precision", type=int, default=32, help="Precision.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs.")
    parser.add_argument("--patience", type=int, default=2, help="How many epochs to wait for improvement in validation.")
    parser.add_argument("--nworkers", type=int, default=mp.cpu_count(), help="Number of workers to use in dataloading.")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Single word describing experiment.")
    parser.add_argument("--description", type=str, default="No description.", help="Single phrase describing experiment.")
    parser.add_argument("--use-enc-attn", action="store_true", help="Use Encoder Attention during decoding")
    parser.add_argument("--same-enc", action="store_true", help="Use separe encoder and decoder or not")
    parser.add_argument("--debug", action="store_true", help="Fast dev run mode.")
    parser.add_argument("--load", type=str, default=None, help="Pre trained model to start with.")
    parser.add_argument("--accelerator", type=str, default='ddp', help="Multiple GPUs Accelerator")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU.")
    
    #uncommend if running through CLI
    hparams = parser.parse_args()
    
    #uncommend if running through Jupyter
#     hparams = parser.parse_args([])
    
    print("Hyperparameters")
    for k, v in vars(hparams).items():
        print(f"{k}: {v}")
    
    if hparams.debug:
        logger = False
        callbacks = None
    else:
        logger = NeptuneLogger(api_key=os.getenv('NEPTUNE_API_TOKEN'),
                               project_name="levy.gurgel/wikitable",
                               experiment_name=hparams.experiment_name,
                               tags=[hparams.description],
                               params=vars(hparams))
        
        dir_path = os.path.join("models", hparams.experiment_name)
        filename = "{epoch}-{val_extact_match:.2f}-{f1_val:.2f}-same"
        callbacks = [ModelCheckpoint(prefix=hparams.experiment_name,
                                     dirpath=dir_path,
                                     filename=filename,
                                     monitor="f1_val",
                                     mode="max")]
    
    model = CNNTransformer(hparams=hparams)

    trainer = pl.Trainer(max_epochs=hparams.max_epochs,
                                 gpus=0 if hparams.cpu else hparams.gpus,
                                 accumulate_grad_batches=hparams.acum,
                                 precision=hparams.precision,
                                 fast_dev_run=hparams.debug,
                                 logger=logger,
                                 callbacks=callbacks,
                                 checkpoint_callback=False if hparams.debug else True,
                                 accelerator=None if hparams.gpus <= 1 else hparams.accelerator
                                 
                        )
    trainer.fit(model)

