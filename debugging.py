
# open the training json file and plot the loss
import json
import matplotlib.pyplot as plt
import numpy as np



def plot_loss_curve(log_file_path):
    with open(log_file_path, 'r') as f:
        log_data = json.load(f)

    steps = []
    losses = []
    for entry in log_data:
        steps.append(entry['step'])
        losses.append(entry['loss'])

    # Convert to numpy arrays for easier manipulation
    steps = np.array(steps)
    losses = np.array(losses)

    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label='Loss', color='blue')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # do this
    log_file_path = "./v1-checkpoints/checkpoint-500.trainer-state.json"
    plot_loss_curve(log_file_path)