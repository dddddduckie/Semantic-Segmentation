import torch.nn as nn
import torch
from model.backbone.layer_factory import shuffle_unit, ConvLayer
from utils.net_util import weight_initialization


class ShuffleNetV2(nn.Module):
    def __init__(self, stage_repeat_nums, stage_out_dims, num_classes, block=shuffle_unit):
        super(ShuffleNetV2, self).__init__()

        assert len(stage_repeat_nums) == 3, "the length of stage_repeat_nums is expected to be 3"
        assert len(stage_out_dims) == 5, "the length of stage_out_dims is expected to be 5"

        in_dim = 3
        out_dim = stage_out_dims[0]
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = []
        self.layer5 = []

        self.layer1 += [ConvLayer(in_dim=in_dim, out_dim=out_dim, kernel_size=3, stride=2, padding=1,
                                  use_bias=False, norm='bn', activation='relu')]
        in_dim = out_dim
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # construct stage 2-4
        stage_names = [2, 3, 4]
        for name, repeat_num, stage_dim in zip(stage_names, stage_repeat_nums, stage_out_dims[1:]):
            cur = []
            cur += [block(in_dim, stage_dim, 2)]
            for i in range(repeat_num - 1):
                cur += [block(stage_dim, stage_dim, 1)]
            in_dim = stage_dim

            if name == 2:
                self.layer2 = cur
            elif name == 3:
                self.layer3 = cur
            elif name == 4:
                self.layer4 = cur

        out_dim = stage_out_dims[-1]
        self.layer5 += [ConvLayer(in_dim=in_dim, out_dim=out_dim, kernel_size=1, stride=1, padding=0,
                                  use_bias=False, norm='bn', activation='relu')]

        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Sequential(*self.layer2)
        self.layer3 = nn.Sequential(*self.layer3)
        self.layer4 = nn.Sequential(*self.layer4)
        self.layer5 = nn.Sequential(*self.layer5)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.max_pool(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.mean([2, 3])
        # out = self.classifier(out)

        return out

def load_pretrained_shufflenet(path, model):
    """
    load a pretrained backbone
    """
    pretrained_dict = torch.load(path)
    pretrained_values = []

    for i in pretrained_dict.values():
        pretrained_values.append(i)

    new_values = pretrained_values
    count = len(new_values)

    new_state_dict = model.state_dict()
    cur = 0

    for k in new_state_dict:
        if cur == count:
            break
        if k.split('.')[-1] == 'num_batches_tracked':
            continue
        else:
            new_state_dict[k] = new_values[cur]
            cur = cur + 1
    model.load_state_dict(new_state_dict)

def shufflenet_v2_1x(pre_trained=True):
    shufflenet = ShuffleNetV2(stage_repeat_nums=[4, 8, 4], stage_out_dims=[24, 116, 232, 464, 1024],
                              num_classes=1000)
    if pre_trained:
        load_pretrained_shufflenet("../pretrained_models/shufflenetv2_x1.pth", shufflenet)
    else:
        shufflenet = weight_initialization(shufflenet)
    return shufflenet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device=torch.device("cpu")
    shufflenet_v2 = shufflenet_v2_1x(pre_trained=True)
    shufflenet_v2 = shufflenet_v2.to(device)
    image = torch.ones(1, 3, 700, 1280)
    image = image.to(device)
    output= shufflenet_v2(image)
    print(output)
    print(output.size())

# model=shufflenet_v2_1x(pre_trained=False)
# for k,v in model.state_dict().items():
#     print(k)