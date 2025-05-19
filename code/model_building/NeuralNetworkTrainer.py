import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetworkTrainer:
    def __init__(self, model, task, lr=0.001, class_weights=None):
        self.model = model
        self.task = task
        self.criterion = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        if task == 'classification':
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def train(self, X_train, y_train, epochs=100, batch_size=32, sampler=None):
        self.model.train()
        dataset = TensorDataset(X_train, y_train)

        if sampler:
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            if self.task == 'classification':
                return torch.argmax(outputs, dim=1)
            else:
                return outputs.squeeze()

    def evaluate(self, X_test, y_test, batch_size=32):
        self.model.eval()
        dataset = TensorDataset(X_test, y_test)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = self.model(X_batch)
                if self.task == 'classification':
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = outputs.squeeze()
                all_preds.append(preds)
                all_labels.append(y_batch)

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)

        if self.task == 'classification':
            print("\n=== Classification Report ===")
            print(classification_report(y_true.numpy(), y_pred.numpy(), zero_division=0))
        else:
            mse = mean_squared_error(y_true.numpy(), y_pred.numpy())
            r2 = r2_score(y_true.numpy(), y_pred.numpy())
            print("\n=== Regression Evaluation ===")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")