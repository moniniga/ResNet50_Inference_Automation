import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as T
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Evaluating ResNet50 model on device: {device}")

model = models.resnet50(pretrained=True).to(device)
model.eval()

transforms = T.Compose([
    T.Resize(size=256, interpolation=T.InterpolationMode.BILINEAR, max_size=None, antialias=None),
    T.CenterCrop(size=(224, 224)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = torchvision.datasets.ImageFolder(root="/home/datasets/imagenet/val", transform=transforms)
data_loader = DataLoader(dataset,batch_size=32, shuffle=False, num_workers=32)

top1_correct_pred = 0
top1_wrong_pred = 0
top5_correct_pred = 0
top5_wrong_pred = 0


for image, target in tqdm.tqdm(data_loader):
    with torch.no_grad():
        image = image.to(device)
        target = target.to(device)
        output = model(image)
    
    # Top1 Accuracy Calculation
    for out,tgt in zip(output,target):
        if out.argmax().item() == tgt.item():
            top1_correct_pred += 1
        else:
            top1_wrong_pred += 1

    # Top5 Accuracy Calculation:
    for out1,tgt1 in zip(output,target):
        if tgt1.item() in torch.topk(out1, 5).indices[1]:
            top5_correct_pred += 1
        else:
            top5_wrong_pred += 1

print(f"Top1: {(top1_correct_pred/(top1_correct_pred + top1_wrong_pred))*100}")
print(f"Top5: {(top5_correct_pred/(top5_correct_pred + top5_wrong_pred))*100}")



