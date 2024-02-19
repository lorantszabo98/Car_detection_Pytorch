import torch
from src.dataset import get_datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_dataset, num_epochs=10):
    # define device to support GPU accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the device
    model.to(device)
    # Set the model to training mode
    model.train()

    # Define data loader and optimizer (change the otimizer and lr to fine tuning)
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=train_dataset.collate_fn, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loop over the epochs
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0

        # Iterate over batches in the training loader
        for images, targets in train_loader:
            # Move images and targets to the device (GPU)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the gradients in the optimizer
            optimizer.zero_grad()
            # Forward pass: compute model predictions and calculate losses
            loss_dict = model(images, targets)
            # Sum all individual losses to get the total loss
            losses = sum(loss for loss in loss_dict.values())
            # Backward pass: compute gradients with respect to model parameters
            losses.backward()
            # Update the model parameters using the optimizer
            optimizer.step()

            # Accumulate the total loss for logging
            total_loss += losses.item()

        # Print average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}")

    # Save the trained model
    torch.save(model.state_dict(), '../trained_models/fasterrcnn_resnet50_fpn_fine_tuned.pth')


if __name__ == "__main__":

    # init the training dataset with the custom class
    trainset, _ = get_datasets()

    # Init the model with pretrained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(preTrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # train the model
    train_model(model, trainset, num_epochs=15)