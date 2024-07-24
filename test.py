from pytorch_ood.model import WideResNet
from pytorch_ood.detector import EnergyBased
from pytorch_ood.utils import OODMetrics

# Create Neural Network
model = WideResNet(num_classes=10, pretrained="er-cifar10-tune").eval().cuda()

# Create detector
detector = EnergyBased(model)

# Evaluate
metrics = OODMetrics()

for x, y in data_loader:
    metrics.update(detector(x.cuda()), y)

print(metrics.compute())