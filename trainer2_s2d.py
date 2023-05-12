import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
# from dataset import KoBARTSummaryDataset
# from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, BartTokenizerFast
# from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartTokenizer, PreTrainedTokenizerFast, BartModel
from torch.utils.data import Dataset, DataLoader, IterableDataset
import logging
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from transformers import BartForConditionalGeneration, BartTokenizer, PreTrainedTokenizerFast, BartModel
# from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
# from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
# from torchtext.data.metrics import bleu_score
# from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

# wandb.init(project='OSSP')
parser = argparse.ArgumentParser(description='AsianBART translation')

parser.add_argument('--checkpoint_path',
                    type=str,
                    default= "D:/OSSP_model_ckpt/asianbart/",
                    help='checkpoint path')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch_size',
                            type=int,
                            default=28,
                            help='')

        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser
    

class DialectDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, pad_index = 0, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t', lineterminator='\n')
        self.len = len(self.docs)
        self.pad_index = pad_index
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['standard'])
        input_ids = self.add_padding_data(input_ids)
        
        label_ids = self.tokenizer.encode(instance['dialect'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        # return (torch.tensor(input_ids, dtype=torch.long),
        #         torch.tensor(dec_input_ids, dtype=torch.long),
        #         torch.tensor(label_ids, dtype=torch.long))
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'decoder_input_ids': torch.tensor(dec_input_ids, dtype=torch.long),
                'labels': torch.tensor(label_ids, dtype=torch.long)}

        # return {'input_ids': np.array(input_ids, dtype=torch.long),
        #         'decoder_input_ids': np.array(dec_input_ids, dtype=torch.long),
        #         'labels': np.array(label_ids, dtype=torch.long)}

class DialectDataModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok,
                 max_len=512,
                 batch_size=8,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = DialectDataset(self.train_file_path,
                                 self.tokenizer,
                                 self.max_len)
        self.test = DialectDataset(self.test_file_path,
                                self.tokenizer,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test

class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=14,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='asianbart model path')
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = 4
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 0
        self.tokenizer = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')

    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()

        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self.log('val_loss', loss)
        return (loss)
        

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss',torch.stack(losses).mean(), prog_bar=True)
        # val_loss = torch.stack([x for x in outputs]).mean()
        # self.log('val_loss', val_loss)

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = DialectDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    model = KoBARTConditionalGeneration(args)

    dm = DialectDataModule('data/train_cleaned.tsv',
                        'data/train_cleaned.tsv',
                        None,
                        max_len=args.max_len,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                        dirpath="D:/OSSP_model_ckpt/asianbart/",
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1,
                                                       prefix='Asianbart_translation',
                                                       )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join("D:/OSSP_model_ckpt/asianbart", "tb_logs"))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)