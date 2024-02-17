import torch
from torch.utils.data import DataLoader
import torch.optim as optim
# from src.model import CarDetectorModel
from tqdm import tqdm


def train_model(model, train_dataset, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Define data loader and optimizer
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=train_dataset.collate_fn, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0

        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        # Print average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}")

    # Save the trained model
    torch.save(model.state_dict(), )
