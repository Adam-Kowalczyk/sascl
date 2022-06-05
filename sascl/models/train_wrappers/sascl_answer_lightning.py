import torch
import torchmetrics
import torch.nn.functional as F

import pytorch_lightning as pl

from sascl.helpers.helpers import expand_dim

class SASCLClassifierLightning(pl.LightningModule):
    def __init__(self, scl, lr=0.01):
        super(SASCLClassifierLightning, self).__init__()
        self.scl = scl
        self.lr = lr
        metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy()})
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def forward(self, questions, answers):
        answers = answers.unsqueeze(2)
        questions = expand_dim(questions, dim=1, k=8)

        permutations = torch.cat((questions, answers), dim=2)
        return self.scl(permutations)

    def training_step(self, batch, batch_index):
        train_questions, train_answers, train_labels = batch

        output = self.forward(train_questions, train_answers)

        loss = F.cross_entropy(output, train_labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        metric_outputs = self.train_metrics(output, train_labels)
        self.log_dict(metric_outputs, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        questions, answers, labels = batch
        y_hat = self.forward(questions, answers)
        loss = F.cross_entropy(y_hat, labels)

        metric_outputs = self.valid_metrics(y_hat, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log_dict(metric_outputs, on_step=False, on_epoch=True)
        return metric_outputs

    def test_step(self, batch, batch_idx):
        questions, answers, labels = batch
        y_hat = self.forward(questions, answers)
        loss = F.cross_entropy(y_hat, labels)
        self.log("test_loss", loss)
        
    def validation_epoch_end(self, validation_step_outputs):
        all_acc = torch.stack([res["val_accuracy"] for res in validation_step_outputs])
        print(f'\n Val Acc: {torch.mean(all_acc).item()}')
            

    def configure_optimizers(self):
        return torch.optim.Adam(self.scl.parameters(), lr=self.lr)