# Only use this script when training models for individual datasets.

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def main():
    """
    Main function to execute the data preparation, model training, evaluation, and prediction.
    """
    data_num = 1  # Change this to the appropriate dataset number
    folder_name = f'Homo Unfiltered New Data {data_num}'  # Change this to the appropriate folder name
    file_prefix = f'homo_unfiltered_new_data_{data_num}'  # Change this to the appropriate file prefix

    def prepare_data_for_model(input_filename, output_filename):
        """
        Prepares data for the model by shifting and modifying columns.

        Args:
            input_filename (str): The input file name containing the raw data.
            output_filename (str): The output file name to save the modified data.
        """
        data = np.loadtxt(input_filename)
        shifted_first_col = np.roll(data[:, 0], -17)
        shifted_first_col[-17:] = 0
        data[:, 0] = shifted_first_col
        new_cols = data[:, -3:]
        shifted_new_cols = np.roll(new_cols, -17, axis=0)
        shifted_new_cols[-17:, :] = 0
        modified_data = np.hstack((data, shifted_new_cols))
        modified_data = modified_data[:-17, :]
        np.savetxt(output_filename, modified_data, fmt='%.16f')
        print(f'Modified data saved to {output_filename}')

    prepare_data_for_model(f'{file_prefix}.txt', f'{file_prefix}_modified.txt')

    def prepare_data_for_graphing(graphing_input, output_filename):
        """
        Prepares data for graphing by adjusting time values and extracting relevant columns.

        Args:
            graphing_input (str): The input file name containing the raw data for graphing.
            output_filename (str): The output file name to save the modified graphing data.

        Returns:
            tuple: A tuple containing the adjusted time values and the extracted columns.
        """
        data = np.loadtxt(graphing_input)
        delta_t_values = data[:, 0]
        t_values = np.copy(delta_t_values)
        num_rows = len(delta_t_values)
        num_groups = num_rows // 17
        cumulative_sum = 0

        for i in range(num_groups):
            start_idx = i * 17
            end_idx = start_idx + 17
            t_values[start_idx:end_idx] += cumulative_sum
            cumulative_sum += delta_t_values[end_idx - 1]

        if num_rows % 17 != 0:
            t_values[num_groups * 17:] += cumulative_sum

        u_macro = data[:, 1]
        v_macro = data[:, 2]
        a_macro = data[:, 3]
        u_micro = data[:, 4]
        v_micro = data[:, 5]
        a_micro = data[:, 6]
        modified_data = np.column_stack((t_values, u_macro, v_macro, a_macro, u_micro, v_micro, a_micro))
        np.savetxt(output_filename, modified_data, fmt='%.16f')
        print(f'Graphing data saved to {output_filename}\n')
        return t_values, u_macro, v_macro, a_macro, u_micro, v_micro, a_micro

    t_values, u_macro, v_macro, a_macro, u_micro, v_micro, a_micro = prepare_data_for_graphing(f'{file_prefix}.txt', f'{file_prefix}_graphing.txt')

    def plot_individual():
        """
        Plots individual graphs for macro and micro displacement, velocity, and acceleration.
        """
        plt.scatter(t_values, u_macro, label="u_macro", alpha=0.3, s=5)
        plt.scatter(t_values, u_micro, label="u_micro", alpha=0.3, s=5)
        plt.xlabel("t")
        plt.ylabel(f"Displacement Values for Dataset {data_num}")
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\full u data {data_num}.png')
        plt.close()

        plt.scatter(t_values, v_macro, label="v_macro", alpha=0.3, s=5)
        plt.scatter(t_values, v_micro, label="v_micro", alpha=0.3, s=5)
        plt.xlabel("t")
        plt.ylabel(f"Velocity Values for Dataset {data_num}")
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\full v data {data_num}.png')
        plt.close()

        plt.scatter(t_values, a_macro, label="a_macro", alpha=0.3, s=5)
        plt.scatter(t_values, a_micro, label="a_micro", alpha=0.3, s=5)
        plt.xlabel("t")
        plt.ylabel(f"Acceleration Values for Dataset {data_num}")
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\full a data {data_num}.png')
        plt.close()

    plot_individual()
    plt.scatter(t_values, u_macro, label="u_macro", alpha=0.3, s=5)
    plt.scatter(t_values, v_macro, label="v_macro", alpha=0.3, s=5)
    plt.scatter(t_values, a_macro, label="a_macro", alpha=0.3, s=5)
    plt.scatter(t_values, u_micro, label="u_micro", alpha=0.3, s=5)
    plt.scatter(t_values, v_micro, label="v_micro", alpha=0.3, s=5)
    plt.scatter(t_values, a_micro, label="a_micro", alpha=0.3, s=5)
    plt.xlabel("t")
    plt.ylabel(f"All Values for Dataset {data_num}")
    plt.legend()
    plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\full data {data_num} (all).png')
    plt.close()

    def load_data_for_model(training_input):
        """
        Loads data for the model from a file.

        Args:
            training_input (str): The input file name containing the training data.

        Returns:
            tuple: A tuple containing the extracted columns from the training data.
        """
        data = np.loadtxt(training_input)
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
        return delta_t, u_macro, v_macro, a_macro, u_micro, v_micro, a_micro, target_u_micro, target_v_micro, target_a_micro

    def export_actual_and_predicted(t_val, actual_vals, predicted_vals, full):
        """
        Exports actual and predicted values to text files.

        Args:
            t_val (numpy.ndarray): The time values.
            actual_vals (numpy.ndarray): The actual values.
            predicted_vals (numpy.ndarray): The predicted values.
            full (bool): Whether to export the full dataset or not.
        """
        if full:
            string = 'full_'
            t_values_col = t_val[:len(predicted_vals)].reshape(-1, 1)
        else:
            string = ''
            t_values_col = np.roll(t_val[:len(predicted_vals)].reshape(-1, 1), -3)

        actual = np.hstack((t_values_col, actual_vals))
        predicted = np.hstack((t_values_col, predicted_vals))
        np.savetxt(f'{file_prefix}_{string}actual_values.txt', actual, delimiter=' ', fmt='%.16f')
        np.savetxt(f'{file_prefix}_{string}predicted_values.txt', predicted, delimiter=' ', fmt='%.16f')

    (delta_t, u_macro, v_macro, a_macro, u_micro, v_micro, a_micro, target_u_micro, target_v_micro, target_a_micro) = load_data_for_model(f'{file_prefix}_modified.txt')

    features = np.column_stack((delta_t, u_macro, v_macro, a_macro, u_micro, v_micro, a_micro))
    targets = np.column_stack((target_u_micro, target_v_micro, target_a_micro))

    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)

    scaler_targets = MinMaxScaler()
    targets_scaled = scaler_targets.fit_transform(targets)

    x_scaled_train, x_scaled_test, y_scaled_train, y_scaled_test = train_test_split(features_scaled, targets_scaled, test_size=0.2, shuffle=False)
    x_train_tensor = torch.tensor(x_scaled_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_scaled_train, dtype=torch.float64)
    x_test_tensor = torch.tensor(x_scaled_test, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_scaled_test, dtype=torch.float64)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel()
    loss_function = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    epochs = 1000  # Adjust as needed
    epoch_patience = 150  # Number of epochs to wait before early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    print("Training the model:")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for seq_batch, label_batch in train_loader:
            optimizer.zero_grad()
            batch_size = seq_batch.size(0)
            model.hidden_cell = (torch.zeros(1, batch_size, model.hidden_layer_size),
                                 torch.zeros(1, batch_size, model.hidden_layer_size))
            seq_batch = seq_batch.view(-1, 1, 7)
            y_pred = model(seq_batch)
            loss = loss_function(y_pred, label_batch)
            loss.backward()
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

        export_actual_and_predicted(t_values, y_test_rescaled, test_predictions_rescaled, False)

        actual_data = np.loadtxt(f'{file_prefix}_actual_values.txt', skiprows=14)
        predicted_data = np.loadtxt(f'{file_prefix}_predicted_values.txt', skiprows=14)

        if len(actual_data) >= 4:
            replacement_value = actual_data[-4, 0]
            actual_data[-3:, 0] = replacement_value

        if len(predicted_data) >= 4:
            replacement_value = predicted_data[-4, 0]
            predicted_data[-3:, 0] = replacement_value

        num_points = 17
        actual_chunks = [actual_data[i:i + num_points] for i in range(0, len(actual_data), num_points) if len(actual_data[i:i + num_points]) == num_points]
        predicted_chunks = [predicted_data[i:i + num_points] for i in range(0, len(predicted_data), num_points) if len(predicted_data[i:i + num_points]) == num_points]
        actual_lines = np.array(actual_chunks).transpose(1, 0, 2)
        predicted_lines = np.array(predicted_chunks).transpose(1, 0, 2)

        plt.figure(figsize=(12, 6))
        for i in range(num_points):
            plt.plot(actual_lines[i, :, 0], actual_lines[i, :, 1], 'bo-', alpha=0.3, label='Actual Micro Displacement' if i == 0 else "")
            plt.plot(predicted_lines[i, :, 0], predicted_lines[i, :, 1], 'ro-', alpha=0.3, label='Predicted Micro Displacement' if i == 0 else "")
        plt.xlabel('t')
        plt.ylabel('Micro Displacement')
        plt.title(f'Actual vs Predicted Micro Displacement for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\u micro data {data_num}.png')
        plt.close()

        indices_to_plot = [0, 3, 6, 9, 12, 15]

        plt.figure(figsize=(12, 6))
        for i in indices_to_plot:
            plt.plot(actual_lines[i, :, 0], actual_lines[i, :, 1], 'bo-', alpha=0.3, label='Actual Micro Displacement' if i == 0 else "")
            plt.plot(predicted_lines[i, :, 0], predicted_lines[i, :, 1], 'ro-', alpha=0.3, label='Predicted Micro Displacement' if i == 0 else "")
        plt.xlabel('t')
        plt.ylabel('Micro Displacement')
        plt.title(f'Actual vs Predicted Micro Displacement (Reduced) for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\u micro data {data_num} (reduced).png')
        plt.close()

        plt.figure(figsize=(12, 6))
        for i in range(num_points):
            plt.plot(actual_lines[i, :, 0], actual_lines[i, :, 2], 'bo-', alpha=0.3, label='Actual Micro Velocity' if i == 0 else "")
            plt.plot(predicted_lines[i, :, 0], predicted_lines[i, :, 2], 'ro-', alpha=0.3, label='Predicted Micro Velocity' if i == 0 else "")
        plt.xlabel('t')
        plt.ylabel('Micro Velocity')
        plt.title(f'Actual vs Predicted Micro Velocity for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\v micro data {data_num}.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        for i in indices_to_plot:
            plt.plot(actual_lines[i, :, 0], actual_lines[i, :, 2], 'bo-', alpha=0.3, label='Actual Micro Velocity' if i == 0 else "")
            plt.plot(predicted_lines[i, :, 0], predicted_lines[i, :, 2], 'ro-', alpha=0.3, label='Predicted Micro Velocity' if i == 0 else "")
        plt.xlabel('t')
        plt.ylabel('Micro Velocity')
        plt.title(f'Actual vs Predicted Micro Velocity (Reduced) for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\v micro data {data_num} (reduced).png')
        plt.close()

        plt.figure(figsize=(12, 6))
        for i in range(num_points):
            plt.plot(actual_lines[i, :, 0], actual_lines[i, :, 3], 'bo-', alpha=0.3, label='Actual Micro Acceleration' if i == 0 else "")
            plt.plot(predicted_lines[i, :, 0], predicted_lines[i, :, 3], 'ro-', alpha=0.3, label='Predicted Micro Acceleration' if i == 0 else "")
        plt.xlabel('t')
        plt.ylabel('Micro Acceleration')
        plt.title(f'Actual vs Predicted Micro Acceleration for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\a micro data {data_num}.png')
        plt.close()

        # Plot reduced micro a
        plt.figure(figsize=(12, 6))
        for i in indices_to_plot:
            plt.plot(actual_lines[i, :, 0], actual_lines[i, :, 3], 'bo-', alpha=0.3,
                     label='Actual Micro Acceleration' if i == 0 else "")
            plt.plot(predicted_lines[i, :, 0], predicted_lines[i, :, 3], 'ro-', alpha=0.3,
                     label='Predicted Micro Acceleration' if i == 0 else "")
        plt.xlabel('t')
        plt.ylabel('Micro Acceleration')
        plt.title(f'Actual vs Predicted Micro Acceleration (Reduced) for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\a micro data {data_num} (reduced).png')
        plt.close()

        # Plot truth graph for u micro
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test_rescaled[:, 0], test_predictions_rescaled[:, 0],
                    label='Predicted Micro Displacement', color='green', alpha=0.3)
        plt.scatter(y_test_rescaled[:, 0], y_test_rescaled[:, 0],
                    label='Actual Micro Displacement', color='blue', alpha=0.3)
        plt.plot([min(y_test_rescaled[:, 0]), max(y_test_rescaled[:, 0])],
                 [min(y_test_rescaled[:, 0]), max(y_test_rescaled[:, 0])], color='red', linestyle='--',
                 label='Actual Micro Displacement')
        plt.xlabel('Actual Micro Displacement')
        plt.ylabel('Predicted Micro Displacement')
        plt.title(f'Actual vs Predicted Micro Displacement for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\truth graph u micro data {data_num}.png')
        plt.close()

        # Plot truth graph for v micro
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test_rescaled[:, 1], test_predictions_rescaled[:, 1],
                    label='Predicted Micro Velocity', color='green', alpha=0.3)
        plt.scatter(y_test_rescaled[:, 1], y_test_rescaled[:, 1],
                    label='Actual Micro Velocity', color='blue', alpha=0.3)
        plt.plot([min(y_test_rescaled[:, 1]), max(y_test_rescaled[:, 1])],
                 [min(y_test_rescaled[:, 1]), max(y_test_rescaled[:, 1])], color='red', linestyle='--',
                 label='Actual Micro Velocity')
        plt.xlabel('Actual Micro Velocity')
        plt.ylabel('Predicted Micro Velocity')
        plt.title(f'Actual vs Predicted Micro Velocity for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\truth graph v micro data {data_num}.png')
        plt.close()

        # Plot truth graph for a micro
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test_rescaled[:, 2], test_predictions_rescaled[:, 2],
                    label='Predicted Micro Acceleration', color='green', alpha=0.3)
        plt.scatter(y_test_rescaled[:, 2], y_test_rescaled[:, 2],
                    label='Actual Micro Acceleration', color='blue', alpha=0.3)
        plt.plot([min(y_test_rescaled[:, 2]), max(y_test_rescaled[:, 2])],
                 [min(y_test_rescaled[:, 2]), max(y_test_rescaled[:, 2])], color='red', linestyle='--',
                 label='Actual Micro Acceleration')
        plt.xlabel('Actual Micro Acceleration')
        plt.ylabel('Predicted Micro Acceleration')
        plt.title(f'Actual vs Predicted Micro Acceleration for Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\truth graph a micro data {data_num}.png')
        plt.close()

        # Convert full dataset to tensor for prediction
        x_scaled_full = np.vstack((x_scaled_train, x_scaled_test))
        x_full_tensor = torch.tensor(x_scaled_full, dtype=torch.float64)

        # Predict the entire dataset
        model.eval()
        with torch.no_grad():
            full_predictions_scaled = model(x_full_tensor.view(-1, 1, x_scaled_full.shape[1])).numpy()
            # Inverse transform predictions back to original y values
            y_full_rescaled = scaler_targets.inverse_transform(full_predictions_scaled)
            y_full_actual_rescaled = scaler_targets.inverse_transform(targets_scaled.reshape(-1, 3))

        export_actual_and_predicted(t_values, y_full_actual_rescaled, y_full_rescaled, True)

        # Plot u micro for Full Dataset
        plt.figure(figsize=(12, 6))
        plt.scatter(t_values[:len(y_full_actual_rescaled)], y_full_actual_rescaled[:, 0], label='Actual Micro Displacement',
                 color='blue', alpha=0.3)
        plt.scatter(t_values[:len(y_full_rescaled)], y_full_rescaled[:, 0], label='Predicted Micro Displacement', color='red', alpha=0.3)
        plt.xlabel('t')
        plt.ylabel('Micro Displacement')
        plt.title(f'Actual vs Predicted Micro Displacement for Full Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\u micro full data {data_num}.png')
        plt.close()

        # Plot truth graph for u micro for Full Dataset
        plt.figure(figsize=(12, 6))
        plt.scatter(y_full_actual_rescaled[:, 0], y_full_rescaled[:, 0],
                    label='Predicted Micro Displacement', color='green', alpha=0.3)
        plt.scatter(y_full_actual_rescaled[:, 0], y_full_actual_rescaled[:, 0],
                    label='Actual Micro Displacement', color='blue', alpha=0.3)
        plt.plot([min(y_full_actual_rescaled[:, 0]), max(y_full_actual_rescaled[:, 0])],
                 [min(y_full_actual_rescaled[:, 0]), max(y_full_actual_rescaled[:, 0])], color='red', linestyle='--',
                 label='Actual Micro Displacement')
        plt.xlabel('Actual Micro Displacement')
        plt.ylabel('Predicted Micro Displacement')
        plt.title(f'Actual vs Predicted Micro Displacement for Full Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\truth graph u micro full data {data_num}.png')
        plt.close()

        # Plot v micro for Full Dataset
        plt.figure(figsize=(12, 6))
        plt.scatter(t_values[:len(y_full_actual_rescaled)], y_full_actual_rescaled[:, 1], label='Actual Micro Velocity',
                 color='blue', alpha=0.3)
        plt.scatter(t_values[:len(y_full_rescaled)], y_full_rescaled[:, 1], label='Predicted Micro Velocity', color='red', alpha=0.3)
        plt.xlabel('t')
        plt.ylabel('Micro Velocity')
        plt.title(f'Actual vs Predicted Micro Velocity for Full Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\v micro full data {data_num}.png')
        plt.close()

        # Plot truth graph for v micro for Full Dataset
        plt.figure(figsize=(12, 6))
        plt.scatter(y_full_actual_rescaled[:, 1], y_full_rescaled[:, 1],
                    label='Predicted Micro Velocity', color='green', alpha=0.3)
        plt.scatter(y_full_actual_rescaled[:, 1], y_full_actual_rescaled[:, 1],
                    label='Actual Micro Velocity', color='blue', alpha=0.3)
        plt.plot([min(y_full_actual_rescaled[:, 1]), max(y_full_actual_rescaled[:, 1])],
                 [min(y_full_actual_rescaled[:, 1]), max(y_full_actual_rescaled[:, 1])], color='red', linestyle='--',
                 label='Actual Micro Velocity')
        plt.xlabel('Actual Micro Velocity')
        plt.ylabel('Predicted Micro Velocity')
        plt.title(f'Actual vs Predicted Micro Velocity for Full Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\truth graph v micro full data {data_num}.png')
        plt.close()

        # Plot a micro for Full Dataset
        plt.figure(figsize=(12, 6))
        plt.scatter(t_values[:len(y_full_actual_rescaled)], y_full_actual_rescaled[:, 2], label='Actual Micro Acceleration',
                 color='blue', alpha=0.3)
        plt.scatter(t_values[:len(y_full_rescaled)], y_full_rescaled[:, 2], label='Predicted Micro Acceleration', color='red', alpha=0.3)
        plt.xlabel('t')
        plt.ylabel('Micro Acceleration')
        plt.title(f'Actual vs Predicted Micro Acceleration for Full Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\a micro full data {data_num}.png')
        plt.close()

        # Plot truth graph for a micro for Full Dataset
        plt.figure(figsize=(12, 6))
        plt.scatter(y_full_actual_rescaled[:, 2], y_full_rescaled[:, 2],
                    label='Predicted Micro Acceleration', color='green', alpha=0.3)
        plt.scatter(y_full_actual_rescaled[:, 2], y_full_actual_rescaled[:, 2],
                    label='Actual Micro Acceleration', color='blue', alpha=0.3)
        plt.plot([min(y_full_actual_rescaled[:, 2]), max(y_full_actual_rescaled[:, 2])],
                 [min(y_full_actual_rescaled[:, 2]), max(y_full_actual_rescaled[:, 2])], color='red', linestyle='--',
                 label='Actual Micro Acceleration')
        plt.xlabel('Actual Micro Acceleration')
        plt.ylabel('Predicted Micro Acceleration')
        plt.title(f'Actual vs Predicted Micro Acceleration for Full Dataset {data_num}')
        plt.legend()
        plt.savefig(f'C:\\Users\\Hirom\\OneDrive\\Vanderbilt\\Research\\Data graphs\\{folder_name}\\truth graph a micro full data {data_num}.png')
        plt.close()


    while True:
        # Get user input for y values and delta_t
        input_delta_t = float(input("Enter the delta t value (can be any positive number): "))
        input_macro_displacement = float(input("Enter the initial macro displacement: "))
        input_macro_velocity = float(input("Enter the initial macro velocity: "))
        input_macro_acceleration = float(input("Enter the initial macro acceleration: "))
        input_micro_displacement = float(input("Enter the initial micro displacement: "))
        input_micro_velocity = float(input("Enter the initial micro velocity: "))
        input_micro_acceleration = float(input("Enter the initial micro acceleration: "))

        # Prepare input data
        input_data = np.array([input_delta_t, input_macro_displacement, input_macro_velocity,
                               input_macro_acceleration, input_micro_displacement, input_micro_velocity,
                               input_micro_acceleration], dtype=np.float64).reshape(1, -1)

        # Scale input data
        input_data_scaled = scaler_features.transform(input_data)

        # Convert to tensor and reshape for LSTM model
        sequence_tensor = torch.tensor(input_data_scaled, dtype=torch.float64).view(1, 1, -1)

        # Model evaluation
        model.eval()
        with torch.no_grad():
            # Make prediction
            future_pred_scaled = model(sequence_tensor).numpy().flatten()

            # Inverse transform to get actual values
            predicted_values = scaler_targets.inverse_transform(
                future_pred_scaled.reshape(-1, 3))  # Ensure to reshape for 3 outputs

            # Print predicted values for displacement, velocity, and acceleration
            print(f"Predicted values for t + {input_delta_t}:")
            print(f"Micro Displacement: {predicted_values[0, 0]}")
            print(f"Micro Velocity: {predicted_values[0, 1]}")
            print(f"Micro Acceleration: {predicted_values[0, 2]}")

        # Continue or exit with input validation
        while True:
            continue_prompt = input("Do you want to predict another value? (yes/no): ").strip().lower()
            if continue_prompt in ['yes', 'no']:
                if continue_prompt == 'yes':
                    break  # Exit the inner loop and continue to the next iteration of the outer loop
                else:
                    print("Exiting...")
                    exit()  # Exits the entire program
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")


class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=128, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 1
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional).to(torch.float64)
        self.dropout = nn.Dropout(p=0.2)
        # For bidirectional, multiply hidden_layer_size by 2 since there are two directions
        self.linear = nn.Linear(hidden_layer_size * 2, output_size).to(torch.float64)

    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size, dtype=torch.float64).to(input_seq.device)
        c0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size, dtype=torch.float64).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions



if __name__ == '__main__':
    main()
