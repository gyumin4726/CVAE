from dataset import train_loader
from model import CVAE

import lightning as L

model = CVAE()

# TODO max_epochs 값 조정 가능
# max_epochs = 10으로 설정
trainer = L.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_dataloaders=train_loader)
trainer.save_checkpoint("./cvae.ckpt")

print(model.device)
