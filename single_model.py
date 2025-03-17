# Use this script when training a model for multiple datasets (Strategy 1).

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.onnx
import joblib

def load_dataset(file):
    """
    Load dataset from a text file.

    @param file: Path to the text file containing the dataset.
    @return: Numpy array of the dataset.
    """
    data = np.loadtxt(file)
    return data

def split_dataset(data, test_size=0.2):
    """
    Split the dataset into training and testing sets.

    @param data: Numpy array of the dataset.
    @param test_size: Proportion of the dataset to include in the test split.
    @return: Tuple containing training data and testing data.
    """
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def prepare_data(data):
    """
    Prepare the features and targets from the dataset.

    @param data: Numpy array of the dataset.
    @return: Tuple containing features and targets as numpy arrays.
    """
    delta_t = data[:, 0]
    u_macro = data[:, 1]
    v_macro = data[:, 2]
    a_macro = data[:, 3]
    u_micro = data[:, 4]
    v_micro = data[:, 5]
    a_micro = data[:, 6]
    target_u_micro = data[:, 7]
    target_v_micro = data[:, 8]
    target_a_micro = data[:, 9]

    features = np.column_stack((delta_t, u_macro, v_macro, a_macro, u_micro, v_micro, a_micro))
    targets = np.column_stack((target_u_micro, target_v_micro, target_a_micro))

    return features, targets

def main():
    """
    Main function to load data, train the LSTM model, and evaluate its performance.
    """
    file_prefix = 'homo_unfiltered_new_'  # Change this to the appropriate file prefix
    folder_name = 'All Data Homo Unfiltered New'  # Change this to the appropriate folder name

    file_list = [f'{file_prefix}data_{i}_modified.txt' for i in range(1, 12)]

    train_features_list = []
    train_targets_list = []
    test_features_list = []
    test_targets_list = []

    for file in file_list:
        data = load_dataset(file)
        train_data, test_data = split_dataset(data)
        train_features, train_targets = prepare_data(train_data)
        test_features, test_targets = prepare_data(test_data)
        train_features_list.append(train_features)
        train_targets_list.append(train_targets)
        test_features_list.append(test_features)
        test_targets_list.append(test_targets)

    train_features = np.vstack(train_features_list)
    train_targets = np.vstack(train_targets_list)
    test_features = np.vstack(test_features_list)
    test_targets = np.vstack(test_targets_list)

    scaler_features = MinMaxScaler()
    train_features_scaled = scaler_features.fit_transform(train_features)
    test_features_scaled = scaler_features.transform(test_features)

    scaler_targets = MinMaxScaler()
    train_targets_scaled = scaler_targets.fit_transform(train_targets)
    test_targets_scaled = scaler_targets.transform(test_targets)

    # Save the scaler objects
    joblib.dump(scaler_features, f'{file_prefix}scaler_features.pkl')
    joblib.dump(scaler_targets, f'{file_prefix}scaler_targets.pkl')

    x_train_tensor = torch.tensor(train_features_scaled, dtype=torch.float64)
    y_train_tensor = torch.tensor(train_targets_scaled, dtype=torch.float64)
    x_test_tensor = torch.tensor(test_features_scaled, dtype=torch.float64)
    y_test_tensor = torch.tensor(test_targets_scaled, dtype=torch.float64)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel()
    loss_function = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 1000
    epoch_patience = 70
    best_val_loss = float('inf')
    patience_counter = 0

    print("Training the model:")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for seq_batch, label_batch in train_loader:
            optimizer.zero_grad()
            seq_batch = seq_batch.view(-1, 1, 7)
            y_pred = model(seq_batch)
            loss = loss_function(y_pred, label_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq_batch, label_batch in test_loader:
                seq_batch = seq_batch.view(-1, 1, 7)
                y_pred = model(seq_batch)
                loss = loss_function(y_pred, label_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3}, Train Loss: {avg_train_loss:10.10f}, Val Loss: {avg_val_loss:10.10f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= epoch_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

    print(f'Final Epoch: {epoch:3}, Loss: {avg_train_loss:10.10f}, Val Loss: {avg_val_loss:10.10f}')

    # Save the trained model
    torch.save(model.state_dict(), f'{file_prefix}single_model.pth')
    # Export the trained model to ONNX format
    dummy_input = torch.randn(1, 1, 7, dtype=torch.float64)
    onnx_file_path = f'{file_prefix}single_model.onnx'
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


    # Evaluation and testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_predictions = []

        for seq_batch, label_batch in test_loader:
            seq_batch = seq_batch.view(-1, 1, 7)
            y_pred = model(seq_batch)
            loss = loss_function(y_pred, label_batch)
            test_loss += loss.item()
            test_predictions.append(y_pred)

        print(f'Test Loss: {test_loss / len(test_loader):10.8f}')

        test_predictions = torch.cat(test_predictions).numpy()
        test_predictions_rescaled = scaler_targets.inverse_transform(test_predictions)
        y_test_rescaled = scaler_targets.inverse_transform(y_test_tensor.numpy().reshape(-1, 3))

        # Plotting results for each dataset
        for i, file in enumerate(file_list):
            start_idx = i * len(test_predictions_rescaled) // len(file_list)
            end_idx = (i + 1) * len(test_predictions_rescaled) // len(file_list)
            graphing_data = np.loadtxt(f'{file_prefix}data_1_graphing.txt')
            # check length of homo_unfiltered_new_data_i_predicted_values.txt for length below! (3591)
            t_values = graphing_data[-3591:, 0] # change

            # Save predictions to .txt file
            predictions_to_save = np.zeros((end_idx - start_idx, 4))
            predictions_to_save[:, 1] = test_predictions_rescaled[start_idx:end_idx, 0]  # Displacement
            predictions_to_save[:, 2] = test_predictions_rescaled[start_idx:end_idx, 1]  # Velocity
            predictions_to_save[:, 3] = test_predictions_rescaled[start_idx:end_idx, 2]  # Acceleration
            np.savetxt(f'{file_prefix}data_{i + 1}_predicted_values_single_model.txt', predictions_to_save, fmt='%.16f')

            plt.figure(figsize=(12, 6))
            plt.scatter(t_values, y_test_rescaled[start_idx:end_idx, 0], label=f'Actual Micro Displacement', color='blue', alpha=0.3)
            plt.scatter(t_values, test_predictions_rescaled[start_idx:end_idx, 0], label=f'Predicted Micro Displacement', color='red', alpha=0.3)
            plt.xlabel('t')
            plt.ylabel('Micro Displacement')
            plt.title(f'Actual vs Predicted Micro Displacement for Dataset {i+1}')
            plt.legend()
            plt.savefig(
                f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\u micro data {i+1}.png')
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.scatter(t_values, y_test_rescaled[start_idx:end_idx, 1], label=f'Actual Micro Velocity', color='blue', alpha=0.3)
            plt.scatter(t_values, test_predictions_rescaled[start_idx:end_idx, 1], label=f'Predicted Micro Velocity', color='red', alpha=0.3)
            plt.xlabel('t')
            plt.ylabel('Micro Velocity')
            plt.title(f'Actual vs Predicted Micro Velocity for Dataset {i+1}')
            plt.legend()
            plt.savefig(
                f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\v micro data {i+1}.png')
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.scatter(t_values, y_test_rescaled[start_idx:end_idx, 2], label=f'Actual Micro Acceleration', color='blue', alpha=0.3)
            plt.scatter(t_values, test_predictions_rescaled[start_idx:end_idx, 2], label=f'Predicted Micro Acceleration', color='red', alpha=0.3)
            plt.xlabel('t')
            plt.ylabel('Micro Acceleration')
            plt.title(f'Actual vs Predicted Micro Acceleration for Dataset {i+1}')
            plt.legend()
            plt.savefig(
                f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\a micro data {i+1}.png')
            plt.close()


class LSTMModel(nn.Module):
    """
    LSTM Model for predicting micro displacement, velocity, and acceleration.

    @param input_size: Number of input features.
    @param hidden_layer_size: Number of hidden units in the LSTM layer.
    @param output_size: Number of output features.
    """
    def __init__(self, input_size=7, hidden_layer_size=64, output_size=3):  # hidden size 32 for non-hetero, 64 for hetero
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 1
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional).to(torch.float64)
        self.dropout = nn.Dropout(p=0.2)  # change: 0.2 for hetero, 0.4 for homo
        # For bidirectional, multiply hidden_layer_size by 2 since there are two directions
        self.linear = nn.Linear(hidden_layer_size * 2, output_size).to(torch.float64)

    def forward(self, input_seq):
        """
        Forward pass through the LSTM model.

        @param input_seq: Input sequence tensor.
        @return: Predictions tensor.
        """
        h0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size, dtype=torch.float64).to(input_seq.device)
        c0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size, dtype=torch.float64).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


if __name__ == '__main__':
    main()