import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from models import get_model
from data_generation import prepare_train_test_data

def train_and_evaluate(stock_dataframes, X, Y, Z, model_type, epochs, batch_size, learning_rate, architecture_params):
    # Prepare data
    data = prepare_train_test_data(stock_dataframes, X, Y, Z)
    features, targets = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model
    model = get_model(model_type, **architecture_params)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test.unsqueeze(1)).item()
        accuracy = ((predictions.round() == y_test.unsqueeze(1)).float().mean()).item()

    return test_loss, accuracy

def grid_search(stock_dataframes, param_grid):
    results = []

    for X in param_grid['X']:
        for Y in param_grid['Y']:
            for Z in param_grid['Z']:
                for model_type in param_grid['model_type']:
                    for epochs in param_grid['epochs']:
                        for architecture_params in param_grid['architecture']:
                            test_loss, accuracy = train_and_evaluate(
                                stock_dataframes, X, Y, Z, model_type, epochs,
                                param_grid['batch_size'], param_grid['learning_rate'], architecture_params
                            )
                            result = {
                                'X': X,
                                'Y': Y,
                                'Z': Z,
                                'model_type': model_type,
                                'epochs': epochs,
                                'test_loss': test_loss,
                                'accuracy': accuracy
                            }
                            results.append(result)
                            print(result)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('training_results.csv', index=False)