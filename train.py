import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DAE


def train_model(device, dataset, epochs, cr, batch_size=64, lr=0.001):
    print(f"--- Training DAE Model (CR={cr}) on {device} ---")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DAE(cr=cr).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_hist = []

    for ep in range(epochs):
        model.train()
        ep_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / len(loader)
        loss_hist.append(avg_loss)

        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep + 1}/{epochs}: Loss {avg_loss:.5f}")

    return model, loss_hist