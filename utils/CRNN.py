import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) for audio-based classification tasks.
    Combines CNN layers for feature extraction and an LSTM for sequence modeling, followed by fully connected layers for classification.
    """
    def __init__(self, input_channels, num_classes, rnn_hidden_size=128, rnn_layers=2, dropout=0.3, device='cpu'):
        """
        Initializes the CRNN model.

        Args:
            input_channels (int): Number of input channels for the CNN.
            num_classes (int): Number of output classes for classification.
            rnn_hidden_size (int, optional): Hidden size of the LSTM. Default is 128.
            rnn_layers (int, optional): Number of LSTM layers. Default is 2.
            dropout (float, optional): Dropout rate for the fully connected layers. Default is 0.3.
            device (str, optional): Device to load the model on ('cpu' or 'cuda'). Default is 'cpu'.
        """

        super(CRNN, self).__init__()
        self.to(device)
        self.device = device
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.rnn = nn.LSTM(
            input_size=1024,  # CNN Output channels * freq_out
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the CRNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_mel_freq_bins, num_frames). Should contain batches of Mel spectrograms.

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_classes).
        """

        # Add mono channel dimension
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, num_mel_freq_bins, num_frames)

        # Pass through CNN layers
        cnn_out = self.cnn(x)  # Shape: (batch_size, 64, frequency_out, time_out)
        
        # Reshape for RNN input
        batch, channels, freq, time = cnn_out.shape
        rnn_input = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)  # Shape: (batch, time_out, 64 * freq_out)
        
        # Pass through RNN layers
        rnn_out, _ = self.rnn(rnn_input)  # Shape: (batch_size, time_out, rnn_hidden_size * 2)
        
        # Take the output of the last valid time step of each sequence for classification
        rnn_out_last = rnn_out[:, -1, :]  # Shape: (batch_size, rnn_hidden_size * 2)
        
        # Pass through fully connected layers
        out = self.fc(rnn_out_last)
        return out

    def train_single_epoch(self, train_dl, loss_fn, optimizer, grad_clip=(True, 1.0), verbose=False):
        """
        Trains the model for a single epoch.

        Args:
            train_dl (DataLoader): DataLoader for the training dataset.
            loss_fn (callable): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            grad_clip (tuple, optional): Tuple indicating whether to clip gradients and the maximum norm. Default is (True, 1.0).
            verbose (bool, optional): Whether to print progress during training. Default is False.

        Returns:
            tuple: Mean loss and mean accuracy over the training epoch.
        """
        self.train(True)

        mean_loss = 0
        mean_accuracy = 0

        for batch, (x, x_og_lengths, y_truth) in enumerate(train_dl):
            x, y_truth = x.to(self.device), y_truth.to(self.device)
            batch_size = len(x)

            # Forward pass
            y_pred = self.forward(x, x_og_lengths)
            loss = loss_fn(y_pred, y_truth)
            with torch.no_grad():
                mean_loss += loss.item()
                accuracy = len(torch.where(y_pred.argmax(-1) == y_truth.argmax(-1))[0]) / batch_size
                mean_accuracy += accuracy

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if grad_clip[0]:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip[1])
            optimizer.step()

            if verbose and batch % 10 == 0:
                print(f"Step [{batch}/{len(train_dl)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}")

        mean_loss /= len(train_dl)
        mean_accuracy /= len(train_dl)
        return mean_loss, mean_accuracy

    def validate(self, val_dl, loss_fn, verbose=False):
        """
        Tests the model on the validation/testing dataset.

        Args:
            val_dl (DataLoader): DataLoader for the validation/testing dataset.
            loss_fn (callable): Loss function.
            verbose (bool, optional): Whether to print progress during validation. Default is False.

        Returns:
            tuple: Mean loss and mean accuracy over the given dataset.
        """
        self.eval()

        mean_loss = 0
        mean_accuracy = 0

        with torch.no_grad():
            for batch, (x, x_og_lengths, y_truth) in enumerate(val_dl):
                x, y_truth = x.to(self.device), y_truth.to(self.device)
                batch_size = len(x)

                # Forward pass
                y_pred = self.forward(x, x_og_lengths)
                loss = loss_fn(y_pred, y_truth)
                mean_loss += loss.item()
                accuracy = len(torch.where(y_pred.argmax(-1) == y_truth.argmax(-1))[0]) / batch_size
                mean_accuracy += accuracy
                top_3 = torch.topk(y_pred, 3, dim=-1).indices
                correct_y = y_truth.argmax(-1)

                if verbose and batch % 10 == 0:
                    print(f"Step [{batch}/{len(val_dl)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}")
                    if batch_size == 1:
                        print(f"Top 3: {top_3[0].tolist()}, Correct class: {correct_y.tolist()}")

        mean_loss /= len(val_dl)
        mean_accuracy /= len(val_dl)
        return mean_loss, mean_accuracy

