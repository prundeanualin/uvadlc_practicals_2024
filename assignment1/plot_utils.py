import matplotlib.pyplot as plt


def plot_train_valid_losses_per_epoch(train_loss, valid_loss):
    if len(train_loss) != len(valid_loss):
        raise ValueError('train_losses and valid_losses must have the same length (=nr of epochs)')

    epochs = list(range(1, len(train_loss) + 1))

    plt.figure(figsize=(12, 7))

    # Plot Training Loss
    plt.plot(epochs, train_loss, marker='o', linestyle='-', linewidth=2, markersize=6,
             label='Training Loss', color='tab:blue')

    # Plot Validation Loss
    plt.plot(epochs, valid_loss, marker='s', linestyle='--', linewidth=2, markersize=6,
             label='Validation Loss', color='tab:orange')

    # Highlight the minimum validation loss
    min_valid_loss = min(valid_loss)
    min_epoch = valid_loss.index(min_valid_loss) + 1
    plt.scatter(min_epoch, min_valid_loss, color='red', s=100, zorder=5)
    plt.annotate(f'Min Val Loss\nEpoch {min_epoch}: {min_valid_loss:.4f}',
                 xy=(min_epoch, min_valid_loss),
                 xytext=(min_epoch, min_valid_loss + 0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12,
                 horizontalalignment='center')

    plt.title('Training and Validation Loss Across Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(epochs)
    plt.legend(title='Loss Type', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('train_valid_losses_per_epoch.png')
    plt.show()


if __name__ == '__main__':
    train_loss = [1.4, 1.2, 0.8, 0.5]
    valid_loss = [l + 0.2 for l in train_loss]
    plot_train_valid_losses_per_epoch(train_loss, valid_loss)