import timm
import torch.nn as nn

__all__ = ["Model"]


class Model(nn.Module):
    def __init__(self, cfg: dict, pretrained: bool = False):
        super(Model, self).__init__()
        self.cfg = cfg
        if cfg["model_type"] == "EfficientNet":
            self.model = timm.create_model(
                self.cfg["model_name"], pretrained=pretrained
            )
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif cfg["model_type"] == "ResNet":
            self.model = timm.create_model(
                self.cfg["model_name"], pretrained=pretrained
            )
            self.n_features = self.model.fc.in_features
        self.fc = nn.Linear(self.n_features, self.cfg["target_size"])

    def feature(self, image):
        feature = self.model(image)
        return feature

    def forward(self, image):
        feature = self.feature(image)
        output = self.fc(feature)
        return output
