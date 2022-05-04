import torch
import os
import numpy as np
import torch.utils.data as data
from model.model_factory import create_model
from parser import create_parser
from torch.backends import cudnn
from dataset.dataloader import create_dataset
from utils.net_util import adjust_learning_rate, create_optimizer
from utils.loss import compute_loss
from utils.file_op import add_summary, sava_checkpoint
from utils.evaluation_metric import confusion_matrix, mIoU
from model.sync_batchnorm.replicate import patch_replication_callback

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_default_tensor_type(torch.FloatTensor)

TORCH_VERSION = torch.__version__
TORCH_CUDA_VERSION = torch.version.cuda
CUDNN_VERSION = str(cudnn.version())
DEVICE_NAME = torch.cuda.get_device_name()

# cudnn.benchmark = True
cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = create_parser()
dataset_name = 'Cityscapes'


def train(model, optimizer, epoch, dataloader):
    num_iters = len(dataloader)  # iters per epoch
    max_iter = parser.max_epoch * num_iters
    model.train()
    model.to(device)

    for i, samples in enumerate(dataloader):
        images, labels, _ = samples
        cur_iter = (epoch - 1) * num_iters + i + 1
        adjust_learning_rate(optimizer=optimizer, cur_iter=cur_iter, ini_lr=parser.learning_rate,
                             step_size=parser.step_size, max_iter=max_iter, mode='poly')
        images = images.to(device)
        labels = labels.long().to(device)

        output = model(images)
        loss = compute_loss(output, labels, name='ce', ignore_index=parser.ignore_label)
        optimizer.zero_grad()
        loss.backward()
        if parser.clip_gradient:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=parser.max_norm)
        optimizer.step()
        print("epoch number: {}/{}, iteration: {}/{}, loss value: {}"
              .format(epoch, parser.max_epoch, i + 1, num_iters, loss.item()))

        if parser.tensorboard:
            saved_scalar = {
                'loss': loss.item()
            }
            add_summary(saved_scalar, cur_iter, parser.log_dir)


def validation(model, dataloader):
    cf_matrix = np.zeros((parser.num_classes,) * 2)
    num_iters = len(dataloader)

    # evaluation mode
    model.eval()
    for i, samples in enumerate(dataloader):
        images, labels, _ = samples
        images = images.to(device)
        labels = labels.cpu().numpy()
        with torch.no_grad():
            output = model(images)
        output = output.data.cpu().numpy()
        preds = np.argmax(output, axis=1)
        cf_matrix += confusion_matrix(preds, labels, num_classes=parser.num_classes)

        if (i + 1) % 100 == 0:
            print("iteration: {}/{}".format(i + 1, num_iters))

    cur_mIoU = round(mIoU(cf_matrix) * 100, 2)
    print("current mIoU on the validation set: " + str(cur_mIoU))

    return cur_mIoU


def main():
    # parameter initialization
    total_epoches = parser.max_epoch
    cur_epoch = 1
    # data_dir = parser.data_dir
    save_dir = parser.ckpt_dir

    # dataset preparation
    train_data = create_dataset(dataset_name=dataset_name, set='train')
    val_data = create_dataset(dataset_name=dataset_name, set='val')
    train_dataloader = data.DataLoader(train_data, batch_size=parser.batch_size, shuffle=True,
                                       num_workers=parser.num_workers, pin_memory=True)
    val_dataloader = data.DataLoader(val_data, batch_size=1, shuffle=False,
                                     num_workers=parser.num_workers, pin_memory=True)

    # create model and optimizer
    model = create_model(num_classes=parser.num_classes, name='DeepLab')
    optimizer = create_optimizer(model.get_optim_params(parser), lr=parser.learning_rate,
                                 momentum=parser.momentum, weight_decay=parser.weight_decay, name="SGD")
    # model = torch.nn.DataParallel(model)
    # patch_replication_callback(model)
    best_mIoU = 0
    best_epoch = 1

    if parser.restore:
        print("loading checkpoint...")
        checkpoint = torch.load(save_dir)
        cur_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_mIoU = checkpoint['best_mIoU']

    print("start training...")
    print("pytorch version: " + TORCH_VERSION + ", cuda version: " + TORCH_CUDA_VERSION +
          ", cudnn version: " + CUDNN_VERSION)
    print("available graphical device: " + DEVICE_NAME)
    os.system("nvidia-smi")

    for epoch in range(cur_epoch, total_epoches + 1):
        print("current epoch:" + str(epoch))
        train(model, optimizer, epoch, train_dataloader)
        print("now start evaluating...")
        cur_mIoU = validation(model, val_dataloader)
        if cur_mIoU > best_mIoU:
            best_epoch = epoch
            best_mIoU = cur_mIoU
            state_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mIoU': best_mIoU
            }
            prefix = "Semantic_Segmentation_" + dataset_name
            sava_checkpoint(state_dict, save_dir, prefix=prefix)

    print("finished training, the best mIoU is: " + str(best_mIoU) + " in epoch " + str(best_epoch))


if __name__ == '__main__':
    main()
