import numpy as np
import torch
import joblib
import torch.nn as nn

def load_model(model_path):
    model = LSTMModel()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler

def prepare_input(data, feature_scaler):
    scaled_data = feature_scaler.transform(data)
    reshaped_data = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])
    return np.array(reshaped_data, dtype=np.float64)

def read_input_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(float, line.strip().split()))
            data.append(row[:7])  # only read first 7 columns
    return np.array(data)

def predict_pth(model, input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data)
        return model(input_tensor).numpy()

def save_predictions(predictions, output_file_path):
    reshaped_predictions = predictions.reshape(17, 3)
    np.savetxt(output_file_path, reshaped_predictions, delimiter=' ', fmt='%.16f')

def main():
    # add "_random" to the end of following 5 file names if using strategy 2
    model_path = 'hetero_single_model_random.pth'
    input_file_path = '0hetero_test_random.txt'
    feature_scaler_path = 'hetero_scaler_features_random.pkl'
    target_scaler_path = 'hetero_scaler_targets_random.pkl'
    output_file_path = '0hetero_predictions_random.txt'

    input_data= read_input_data(input_file_path)

    feature_scaler = load_scaler(feature_scaler_path)
    target_scaler = load_scaler(target_scaler_path)

    input_data = prepare_input(input_data, feature_scaler)

    model = load_model(model_path)

    predictions = predict_pth(model, input_data)

    predictions = target_scaler.inverse_transform(predictions)
    save_predictions(predictions, output_file_path)

class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=64, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = 1
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=self.bidirectional).to(torch.float64)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size).to(torch.float64)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_layer_size, dtype=torch.float64).to(input_seq.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_layer_size, dtype=torch.float64).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

if __name__ == '__main__':
    main()