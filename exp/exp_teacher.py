import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from models.teacher import TeacherModel
import os


class Exp_Teacher:

    def __init__(self, args):

        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TeacherModel(args).to(self.device)

    def train(self):

        train_data, train_loader = data_provider(self.args, "train")
        vali_data, vali_loader = data_provider(self.args, "val")

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs):

            self.model.train()

            for batch_x, batch_y, _, _ in train_loader:

                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)

                pred, G = self.model(batch_x)

                loss_forecast = criterion(pred, batch_y)

                loss_sparse = torch.mean(torch.abs(G))

                loss = loss_forecast + 0.001 * loss_sparse

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            print("epoch", epoch, "loss", loss.item())

        torch.save(self.model.state_dict(), "teacher.pth")

        print("Teacher saved")