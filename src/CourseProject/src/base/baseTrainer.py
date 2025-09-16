import torch
import os
import data as datalib
import numpy as np
from tqdm import tqdm
from CONFIG import CONFIG
import logging
from utils.logger import log_function


class baseTrainer:
    
    
    def __init__(self) -> None:
        
        self.cfg = CONFIG 
        self.model_path = None
        utils.create_direcotry(self.model_path)
        tboard_logs_path = os.path.join(self.cfg['name',"tboarrd_logs",...])
        utils.create_direcotry(tboard_logs_path)
        
        
        self.training_losses = []
        self.validation_losses = []
        self.writer = utils.TensorboardWriter(logdir=tboard_logs_path)
        
        self.setup_model()
        return
    
    
    def load_data(self): 
        
        """
        Loading dataset and data-loaders
        """

        self.train_loader  = datalib.build_data_loader(split='train')
        
        self.valid_loader = datalib.build_data_loader(split='validation')
        
        return
    
    
    def setup_model(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        optimizer, scheduler, lr_warmup, criterion = None
        
        model = None
        
        self.model = model
        self.optimizer, self.scheduler, self.num_epochs = optimizer, scheduler, self.cfg['num_epochs']
        self.criterion = criterion
        # self.loss_tracker = loss_tracker
        # self.warmup_scheduler = WarmupVSScehdule(
        #         optimizer=self.optimizer,
        #         lr_warmup=lr_warmup,
        #         scheduler=scheduler
        #     )
        return
    
        
    def train_epoch(self):
        """ Training a model for one epoch """
        
        epoch = self.epoch
        
        # loss_list = []
        for (images, labels) in tqdm(self.train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()
            
            # Forward pass to get output/logits
            outputs = self.model(images)
            
            # Calculate Loss: softmax --> cross entropy loss
            loss = self.criterion(outputs, labels)
            self.training_losses.append(loss.item())
            
            # Getting gradients w.r.t. parameters
            loss.backward()
            
            # Updating parameters
            self.optimizer.step()
            
        mean_loss = np.mean(self.training_losses)
        return mean_loss, self.training_losses


    @torch.no_grad()
    def eval_model(self):
        """ Evaluating the model for either validation or test """
        correct = 0
        total = 0
        # loss_list = []
        
        for images, labels in self.eval_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass only to get logits/output
            outputs = self.model(images)
                    
            loss = self.criterion(outputs, labels)
            self.validation_losses.append(loss.item())
                
            # Get predictions from the maximum value
            preds = torch.argmax(outputs, dim=1)
            correct += len( torch.where(preds==labels)[0] )
            total += len(labels)
                    
        # Total correct predictions and loss
        accuracy = correct / total * 100
        loss = np.mean(self.validation_losses)
        
        return accuracy, loss


    def train_model(self, start_epoch=0):
        """ Training a model for a given number of epochs"""
        
        train_loss = []
        val_loss =  []
        loss_iters = []
        valid_acc = []


        for epoch in range(self.num_epochs):
            # print(f"Started Epoch {epoch+1}/{num_epochs}...")
            logging.info(f"Epoch {epoch+1}/{self.num_epochs}...")
            
            # validation epoch
            logging.info("  --> Running valdiation epoch")
            self.model.eval()  # important for dropout and batch norms
            accuracy, loss = self.eval_model()
            
            valid_acc.append(accuracy)
            val_loss.append(loss)
            # tboard.add_scalar(f'Accuracy/Valid', accuracy, global_step=epoch+start_epoch)
            # tboard.add_scalar(f'Loss/Valid', loss, global_step=epoch+start_epoch)
            self.writer.add_scalar(f'Accuracy/Valid', accuracy, global_step=epoch+start_epoch)
            self.writer.add_scalar(f'Loss/Valid', loss, global_step=epoch+start_epoch)
            
            # training epoch
            logging.info("  --> Running train epoch")
            self.model.train()  # important for dropout and batch norms
            mean_loss, cur_loss_iters = self.train_epoch()
            
            self.scheduler.step()
            train_loss.append(mean_loss)
            self.writer.add_scalar(f'Loss/Train', mean_loss, global_step=epoch+start_epoch)

            loss_iters = loss_iters + cur_loss_iters
            
            logging.info(f"Train loss: {round(mean_loss, 5)}")
            logging.info(f"Valid loss: {round(loss, 5)}")
            logging.info(f"Valid Accuracy: {accuracy}%")
            logging.info("\n")
        
        logging.info(f"Training completed")
        
        # logging.info("Saving final checkpoint")
        # self.wrapper_save_checkpoint(epoch=epoch, finished=True)
        
        # def wrapper_save_checkpoint(self, epoch=None, savedir="models", savename=None, finished=False):
        #     """
        #     Wrapper for saving a models in a more convenient manner
        #     """
        #     setup_model.save_checkpoint(
        #             model=self.model,
        #             optimizer=self.optimizer,
        #             scheduler=self.warmup_scheduler.scheduler,
        #             lr_warmup=self.warmup_scheduler.lr_warmup,
        #             epoch=epoch,
        #             exp_path=self.exp_path,
        #             savedir=savedir,
        #             savename=savename,
        #             finished=finished
        #         )
        #     return
        
        return train_loss, val_loss, loss_iters, valid_acc