import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
from sklearn.utils import compute_class_weight  # This might not be used if class_weights are passed directly as tensor
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetworkTrainer:
    def __init__(self, model, task, lr=0.001, class_weights=None):
        self.model = model
        self.task = task
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Initialize criterion based on task type
        if task == 'classification':
            # Ensure class_weights is a tensor if used with CrossEntropyLoss
            if class_weights is not None:
                # If class_weights is a numpy array, convert it to a torch tensor
                if isinstance(class_weights, torch.Tensor):
                    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                else:  # Assuming class_weights is a list or numpy array
                    self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("Task must be 'regression' or 'classification'")

    def train(self, X_train, y_train, epochs=100, batch_size=32, sampler=None):
        self.model.train()  # Set the model to training mode
        dataset = TensorDataset(X_train, y_train)

        # IMPORTANT FIX: Add drop_last=True to DataLoader
        if sampler:
            # If using a custom sampler, ensure it can handle dropping last batch if needed
            # For typical use-cases, drop_last=True is generally safe.
            loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        else:
            # This is the most common path; ensure drop_last=True here
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # KEY CHANGE HERE

        print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()  # Zero the gradients before each batch
                outputs = self.model(X_batch)  # Forward pass

                # Reshape y_batch for regression if it's not already [batch_size, 1]
                if self.task == 'regression' and outputs.shape != y_batch.shape:
                    y_batch = y_batch.view_as(outputs)

                loss = self.criterion(outputs, y_batch)  # Calculate loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update model weights

            # Print loss every 10 epochs (or adjust frequency)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
        print("Training finished.")

    def predict(self, X):
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            outputs = self.model(X)
            if self.task == 'classification':
                # For classification, return the predicted class index
                return torch.argmax(outputs, dim=1)
            else:
                # For regression, ensure the output matches the expected shape (e.g., flatten if needed)
                return outputs.squeeze()  # Removes dimensions of size 1 (e.g., [N, 1] -> [N])

    def evaluate(self, X_test, y_test, batch_size=32):
        self.model.eval()  # Set model to evaluation mode
        dataset = TensorDataset(X_test, y_test)
        # For evaluation, drop_last is not typically necessary unless you have specific batch requirements
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = self.model(X_batch)
                if self.task == 'classification':
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = outputs.squeeze()  # Match predict output format
                all_preds.append(preds)
                all_labels.append(y_batch)

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)

        if self.task == 'classification':
            print("\n=== Classification Report ===")
            # Ensure y_true and y_pred are numpy arrays for sklearn metrics
            print(classification_report(y_true.cpu().numpy(), y_pred.cpu().numpy(), zero_division=0))
        elif self.task == 'regression':
            # Ensure y_true and y_pred are numpy arrays for sklearn metrics
            mse = mean_squared_error(y_true.cpu().numpy(), y_pred.cpu().numpy())
            r2 = r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
            print("\n=== Regression Evaluation ===")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")