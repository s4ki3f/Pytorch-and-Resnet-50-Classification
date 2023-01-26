# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)

stds = [0.229, 0.224, 0.225]
resizing_ratio = 32
means = [0.485, 0.456, 0.406]

preprocess = transforms.Compose([
    transforms.Resize(resizing_ratio),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds),
])

zer = 0
input_tensor= preprocess(input_image)
input_batch = input_tensor.unsqueeze(zer) 

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    newmodel.to('cuda')

with torch.no_grad():
    output = newmodel(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
