import torch
import numpy as np
from model import CVAE
from dataset import test_loader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import gradio as gr

model = CVAE.load_from_checkpoint("./cvae.ckpt")
model.to(torch.device("cpu"))

print(model.device)


label_mean    = torch.zeros(size=(10,2), dtype=torch.float32)
label_log_var = torch.zeros(size=(10,2), dtype=torch.float32)
label_bool    = np.zeros(shape=(10), dtype=bool)

for step, (x, c) in enumerate(test_loader):
    mean, log_var = model.encoder(x, c)

    for idx, cc in enumerate(c):
        label = int(cc)              # ex) tensor(3) -> 3        
        label_mean[label] = mean[idx]   
        label_log_var[label]  = log_var[idx] 
        label_bool[label] = True

    if (np.all(label_bool)):
        break

label_std = label_log_var.exp().pow(0.5)
# print("label_mean", label_mean)
# print("cls_std", label)


def generate_image(cls, z0, z1):    
    # z = label_mean[cls] + label_std[cls] * torch.tensor([z0, z1])
    z = label_mean[cls] + torch.tensor([z0, z1])
    c = torch.tensor(cls, dtype=torch.int64)
    
    # make a single batch
    z = torch.unsqueeze(z, 0)
    c = torch.unsqueeze(c, 0)

    x_pred = model.decoder(z, c)
    
    #추가
    img_tensor = x_pred[0].view(1, 28, 28)
    #변경경
    gen_img = to_pil_image(img_tensor)

    return gen_img


with gr.Blocks() as demo:
    gr.Markdown("# Generative Model with Conditional Variational AutoEncoder")
    with gr.Row(equal_height=True):
        cls = gr.Number(value=0, minimum=0, maximum=9)
        z0 = gr.Slider(minimum=-10.0, maximum=10.0, value=0.0, step=0.01, label="z0")
        z1 = gr.Slider(minimum=-10.0, maximum=10.0, value=0.0, step=0.01, label="z1")
        button = gr.Button("Generate", variant="primary")
    with gr.Row(equal_height=True):
        gen_image = gr.Image(height=250, width=250)
    
    button.click(
        generate_image, 
        inputs  = [cls, z0, z1],
        outputs = [gen_image],
    )
    

# 실행
demo.launch()
