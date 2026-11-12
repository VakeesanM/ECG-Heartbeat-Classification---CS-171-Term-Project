import torch 
import torch.nn as nn
from torchmetrics import Recall, F1Score
from abc import ABC, abstractmethod
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseModel(nn.Module, ABC): 
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.to(DEVICE)
    @abstractmethod
    def forward(self, X):
        pass

    def start_timing(self):
        self.start_time = time.time()
            
            
    def end_timing(self):
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def train_with_metrics(self, data, epochs=5):
        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        if not self.loss_fn:
            self.loss_fn = nn.CrossEntropyLoss()

        self.start_timing()

        recall_metric = Recall(task="multiclass", num_classes=5, average="macro").to(DEVICE)
        f1_metric = F1Score(task="multiclass", num_classes=5, average="macro").to(DEVICE)

        loss_history = []
        recall_history = []
        f1_score_history = []

        for i in range(epochs):
            self.train()
            train = 0
            recall_metric.reset()
            f1_metric.reset()
            for seq, label in data:
                seq = seq.to(DEVICE)
                label = label.to(DEVICE)
                y_pred = self(seq)
                loss = self.loss_fn(y_pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train += loss.item()
                recall_metric.update(y_pred.detach(), label)
                f1_metric.update(y_pred.detach(), label)
                
            loss_history.append(train / len(data))
            recall_history.append(recall_metric.compute().item())
            f1_score_history.append(f1_metric.compute().item())

            print(f"Epoch #{i+1} | Loss: {loss_history[-1]} | Recall: {(recall_history[-1]*100):.2f}% | F1 Score: {(f1_score_history[-1]*100):.2f}%")

        print(f'Trained Model for {self.end_timing()}')

        return loss_history, recall_history, f1_score_history


    def eval_on_test(self, data):
        if not self.loss_fn:
            self.loss_fn = nn.CrossEntropyLoss()
        recall_metric = Recall(task="multiclass", num_classes=5, average="macro").to(DEVICE)
        f1_metric = F1Score(task="multiclass", num_classes=5, average="macro").to(DEVICE)
        test_loss = 0
        for seq, label in data:
            seq = seq.to(DEVICE)
            label = label.to(DEVICE)

            y_pred = self(seq)

            loss = self.loss_fn(y_pred, label)
            test_loss += loss.item()
            recall_metric.update(y_pred.detach(), label)
            f1_metric.update(y_pred.detach(), label)

        test_loss = test_loss / len(data)
        recall = recall_metric.compute().item()
        f1_Score = f1_metric.compute().item()
        print(f"Evalutation Results | Test Loss: {test_loss} | Recall: {(recall*100):.2f}% | F1 Score: {(f1_Score*100):.2f}% ")

        return test_loss, recall, f1_Score

        



