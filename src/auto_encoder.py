import torch
from torch import nn, optim


class Autoencoder(nn.Module):
    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 1028),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1028, 768),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(768, enc_shape),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 768),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(768, 1028),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1028, in_shape),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class AutoEncoderEmbedding:
    def __init__(self, input_shape, device) -> None:

        self.input_shape = input_shape
        self.device = device
        self.error = nn.MSELoss()

    def train_model(self, n_epochs, x):
        self.model.train()
        for epoch in range(1, n_epochs + 1):
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.error(output, x)
            loss.backward()
            self.optimizer.step()

            # if epoch % int(0.1 * n_epochs) == 0:
            print(f"epoch {epoch} \t Loss: {loss.item():.4g}")

    def embed(self, features):
        with torch.no_grad():
            encoded = self.model.encode(features)
            # decoded = self.model.decode(encoded)
            # mse = self.error(decoded, features).item()
            enc = encoded.cpu().detach().numpy()
            # dec = decoded.cpu().detach().numpy()
        return enc

    def train_embeddings(self, features, n_epochs, save_path=""):
        self.model = (
            Autoencoder(in_shape=self.input_shape, enc_shape=768)
            .double()
            .to(self.device)
        )

        self.optimizer = optim.Adam(self.model.parameters())

        self.train_model(n_epochs=n_epochs, x=features)
        if save_path:
            torch.save(self.model.state_dict(), save_path)

    def load_model(self, path):
        self.model = (
            Autoencoder(in_shape=self.input_shape, enc_shape=768)
            .double()
            .to(self.device)
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == "__main__":
    import src.config
    import pickle
    import json

    with open(src.config.DATA_PREPROCESSED / "heading_meta_data.json", "r") as outfile:
        heading_meta_data = json.load(outfile)

    from sklearn import preprocessing

    lb = preprocessing.LabelBinarizer()

    lb.fit(list(heading_meta_data.keys()))
    heading_embedding = lb.transform(lb.classes_)

    print("done")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # with open(src.config.DATA_PREPROCESSED / "heading_embedding.pkl", "rb") as fOut:
    #     heading_embedding = pickle.load(fOut)["embeddings"][:100]

    train_features = torch.tensor(heading_embedding, dtype=torch.double).to(device)
    # features = torch.tensor(train_features.to_numpy(), dtype=torch.double)
    autoenc_emb = AutoEncoderEmbedding(
        input_shape=train_features.shape[-1], device=device
    )
    # autoenc_emb.load_model(path=src.config.DATA_ROOT / "autoencoder.pth")
    autoenc_emb.train_embeddings(features=train_features, n_epochs=3, save_path="")
    features = autoenc_emb.embed(train_features)
    import pdb

    pdb.set_trace()
    print(features.shape)
    with open(
        src.config.DATA_PREPROCESSED / "heading_embedding_auto_enc.pkl", "wb"
    ) as f:
        pickle.dump(features, f)
