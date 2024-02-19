import torch
from src.dataset import get_datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_dataset, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Define data loader and optimizer
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn, shuffle=True)
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
    torch.save(model.state_dict(), '../trained_models/fasterrcnn_resnet50_fpn_fine_tuned.pth')


if __name__ == "__main__":

    trainset, _ = get_datasets()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(preTrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    train_model(model, trainset, num_epochs=15)