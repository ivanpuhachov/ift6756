import torch
from inception_score import PretrainedInception
from torchvision.models.inception import inception_v3

device = 'cuda'
a = torch.rand(size=(100,3,32,32)).to(device)
pretrained = PretrainedInception().to(device)
with torch.no_grad():
    b = pretrained.forward_no_grad(a)
    c = pretrained.get_activations(a)
    d = pretrained.compute_frechet_stats(a)
print(b.shape)
print(c.shape)
print(d[0].shape)

# inception_model = inception_v3(pretrained=True, transform_input=False)
# inception_model.eval()
# up = torch.nn.Upsample(size=(299, 299), mode='bilinear')
#
# def get_pred(x):
#     with torch.no_grad():
#         x = up(x)
#         print(x.shape)
#         x = inception_model(x)
#     return torch.softmax(x, dim=-1).data.cpu().numpy()
#
# a = torch.rand(size=(10,3,32,32)).to('cpu')
# b = get_pred(a)
# print(b.shape)
# pretrained = PretrainedInception().to(device)
# b = pretrained(a)
# print(b.shape)