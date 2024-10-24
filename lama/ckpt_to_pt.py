import os
import yaml
from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf
import torch
lama_model_path = 'E:/study/python/Study-myself/person-anonymous/lama/big-lama/'

train_config_path = os.path.join(lama_model_path, 'config.yaml')
with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

train_config.training_model.predict_only = True
train_config.visualizer.kind = 'noop'

checkpoint_path = os.path.join(lama_model_path,
                                'models',
                                'best.ckpt')
model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
model.freeze()
with torch.no_grad():
    typical_input = torch.zeros([1, 4, 512, 512])
    # print(model.generator(typical_input).shape)
    traced_cell = torch.jit.trace(model.generator, (typical_input))
torch.jit.save(traced_cell, os.path.join(lama_model_path, 'lama-model-best.pt'))