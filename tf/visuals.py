import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

def plot_history(history):
    """
    Plots the training and validation accuracy and loss from a Keras history object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'],     label='Train',      linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'],     label='Train',      linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_history_with_fine_tuning(history,history_ft):
    """
    Plots the training and validation accuracy and loss from a Keras history object.
    Inspired by https://www.tensorflow.org/tutorials/images/transfer_learning
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    fine_tuning_epoch_start = history.epoch[-1]

    # Merge the training and validation accuracy for the frozen training, followed by the fine-tuning
    acc = history.history['accuracy'].copy()
    acc += history_ft.history['accuracy'].copy()

    acc_val = history.history['val_accuracy'].copy()
    acc_val += history_ft.history['val_accuracy'].copy()

    # Merge the training and validation loss for the frozen training, followed by the fine-tuning
    loss = history.history['loss'].copy()
    loss += history_ft.history['loss'].copy()

    loss_val = history.history['val_loss'].copy()
    loss_val += history_ft.history['val_loss'].copy()
    
    # Accuracy
    ax1.plot(acc,     label='Train',      linewidth=2)
    ax1.plot(acc_val, label='Validation', linewidth=2, linestyle='--')
    ax1.plot([fine_tuning_epoch_start,fine_tuning_epoch_start],[min(min(acc),min(acc_val)),max(max(acc),max(acc_val))], label='Start Fine Tuning')    
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(loss,     label='Train',      linewidth=2)
    ax2.plot(loss_val, label='Validation', linewidth=2, linestyle='--')
    ax2.plot([fine_tuning_epoch_start,fine_tuning_epoch_start],[min(min(loss),min(loss_val)),max(max(loss),max(loss_val))], label='Start Fine Tuning')  
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(all_labels, all_preds, labels, classes, title='Validation Dataset - Confusion matrix'):
    """
    Plots a normalized confusion matrix with F1, Precision, and Recall scores.
    """
    score = f1_score(all_labels, all_preds, labels=labels, average='macro')
    precision = precision_score(all_labels, all_preds, labels=labels, average='macro')
    recall = recall_score(all_labels, all_preds, labels=labels, average='macro')

    cm = confusion_matrix(all_labels, all_preds)
    cm = (cm.T / cm.sum(axis=1)).T # normalize

    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True, cmap='Blues', include_values=False)

    ax = plt.gca()
    ax.set_title(title, fontsize=25, pad=50)
    ax.text(
        0.5, 1.02,   # x=center, y just below title
        f'F1: {score:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}',
        fontsize=18,
        ha='center', va='bottom',
        transform=ax.transAxes
    )
    plt.tight_layout()
    plt.show()
