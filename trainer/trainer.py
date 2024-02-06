import numpy as np
import logging
import torch
from torch.utils.data import DataLoader

from utils import utils, signal_utils, file_utils
import trainer.loss as Loss

# A logger for this file
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, datasets, model_cfg):
        # datasets: dict of keys = {"train", "validation", "test"}
        self.model = model.cuda() if torch.cuda.is_available() else model
        log.info(model.__class__)
        self.optimizer = self.create_optimizer(model_cfg.train.optimizer)

        self.datasets = datasets
        self.model_cfg = model_cfg
        
    def _get_dataloader(self, key, dataloader_cfg):
        if self.datasets.get(key) is None:
            raise ValueError(f"Trainer: missing datasets[{key}].")
        return DataLoader(dataset=self.datasets[key],
            batch_size=dataloader_cfg.batch_size,
            shuffle=dataloader_cfg.shuffle,
            num_workers=dataloader_cfg.num_workers
        )
        

    def create_optimizer(self, optimizer_cfg):
        trainable_params = [p for n, p in self.model.named_parameters()]
        return torch.optim.Adam(trainable_params, lr = optimizer_cfg.lr)

    def create_scheduler(self):
        # add your scheduler code here
        pass

    def compute_loss(self, y_pred, y, loss_cfg):
        return Loss.get_loss(y_pred=y_pred, y=y, loss_cfg=loss_cfg)

    #@utils.timeit
    def training_step(self, dataloader):
        loss_step, n = 0.0, 0
        y_pred_y_arr = []

        self.model.train()# Optional when not using Model Specific layer
        
        for x, y in dataloader:
            n += 1
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            self.optimizer.zero_grad()
            y_pred = self.model(x=x)

            loss = self.compute_loss(y=y, y_pred=y_pred, loss_cfg=self.model_cfg.loss) #(batch_size,)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            loss_step += loss.item()

            y_pred_y_arr.extend(
                    np.hstack((y_pred.detach().cpu().numpy()[:, np.newaxis, :],
                            y.detach().cpu().numpy()[:, np.newaxis, :])))
        loss_step /= n
        y_pred_y_arr = np.array(y_pred_y_arr)

        return loss_step, y_pred_y_arr

    def evaluate_step(self, dataloader):
        '''
            Returns:
                loss: float, (1,), average loss
                xhat: numpy array, (n, window_size), model output
        '''
        loss_step, n = 0.0, 0
        y_pred_y_arr = []
        
        self.model.eval()# Optional when not using Model Specific layer
        for x, y in dataloader:
            n += len(x)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                y_pred = self.model(x=x)
                loss = self.compute_loss(y=y, y_pred=y_pred, loss_cfg=self.model_cfg.loss) #(batch_size,)
                loss = loss.mean()
                loss_step += loss.sum().item() #mean later over all loss can reduce error in float-point

                y_pred_y_arr.extend(
                    np.hstack((y_pred.detach().cpu().numpy()[:, np.newaxis, :],
                            y.detach().cpu().numpy()[:, np.newaxis, :])))
        loss_step /= n
        y_pred_y_arr = np.array(y_pred_y_arr)
        
        return loss_step, y_pred_y_arr

    def train(self):
        cfg = self.model_cfg.train

        # Get dataloaders from datasets
        train_dataloader = self._get_dataloader("train", cfg.dataloader)
        if cfg.do_evaluate:
            eval_dataloader = self._get_dataloader("validation", self.model_cfg.evaluate.dataloader)

        # Load model
        if cfg.checkpoint.do_load:
            self._load_checkpoint(cfg.checkpoint.load_path)
        
        # Per epoch: train, [eval], [save_checkpoint]
        min_loss = np.inf
        for e in range(cfg.n_epoch):
            # Training step
            loss, outputs = self.training_step(train_dataloader)
            msg = f'🍺 Epoch {e+1}/{cfg.n_epoch} \t Training Loss: {loss}'
            # utils.log_accuracy(outputs)
            
            # Validation step
            if cfg.do_evaluate:
                eval_loss, outputs = self.evaluate_step(eval_dataloader)
                msg += f'\t Validation Loss: {eval_loss}'
                if cfg.anchor_eval_loss: loss = eval_loss
                # utils.log_accuracy(outputs)

            log.info(msg)
            
            # Save checkpoint
            if cfg.checkpoint.do_save and min_loss > loss: # Save if loss goes down
                self._save_checkpoint(loss_old=min_loss, loss_new=loss, path=cfg.checkpoint.save_path)
                min_loss = loss

    def predict(self):
        cfg = self.model_cfg.predict
        dataloader = self._get_dataloader("test", cfg.dataloader)
        # Load model
        if cfg.checkpoint.do_load:
            self._load_checkpoint(cfg.checkpoint)
        
        # Evaluate
        loss, outputs = self.evaluate_step(dataloader)
        log.info("🍺 Prediction loss = {}, outputs = {}".format(loss, outputs.shape))
        if cfg.output.do_save:
            path = utils.get_hydra_output_path(cfg.output.path)
            file_utils.dumpPkl(outputs, path)
        # utils.log_accuracy(outputs)

    def _load_checkpoint(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
        log.info("Loaded {}".format(model_path))

    def _save_checkpoint(self, loss_old, loss_new, path):
        # Saving State Dict
        log.info(f'Loss Decreased({loss_old:.6f}--->{loss_new:.6f}) \t Saving the Model in {path}')
        torch.save(self.model.state_dict(), path)

    
