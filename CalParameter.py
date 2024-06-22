"""
CalParameter -

Author:霍畅
Date:2024/6/19
"""
import timm
import torchvision.models as models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# 自定义模型参数
embed_dim = 256  # 增加嵌入维度
num_layers = 13  # 增加Transformer层数
num_heads = 8    # 增加注意力头数量
mlp_ratio = 4    # MLP比率保持不变
img_size = 224   # 输入图像大小

# 创建自定义DeiT模型
model_cfg = {
    'img_size': img_size,
    'embed_dim': embed_dim,
    'depth': num_layers,
    'num_heads': num_heads,
    'mlp_ratio': mlp_ratio
}

# 加载DeiT-Tiny模型
model = timm.create_model('deit_tiny_patch16_224', **model_cfg)
print(model)
parameters01 = count_parameters(model)
print(f"Modified TiV {parameters01/10e5:.2f}M")
# 加载ResNet模型
model = models.resnet18()
print(model)
parameters02 = count_parameters(model)
print(f"ResNet-18 {parameters02/10e5:.2f}M")

print(f"Diff = {(parameters02-parameters01)/parameters02:.2}%")
