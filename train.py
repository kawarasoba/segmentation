import torch
import numpy as np
from absl import flags, app
import torch.nn as nn
import torch.optim as optim
from os import makedirs
import os.path
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

try:
    from apex import amp, optimizers
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from dataset import load_data
import augmentations as aug
import models
from lr_scheduler import cosine_annealing

flags.DEFINE_string('case', None, 'set case.', short_name='c')
flags.DEFINE_string('model_name', None, 'set model_name.', short_name='m')
flags.DEFINE_integer('final_epoch', 50, 'set final_epoch.', short_name='e')
flags.DEFINE_string('train_data_path', 'coco/images', 'set train_images_path.')
flags.DEFINE_string('valid_data_path', 'coco/images', 'set train_images_path.')
flags.DEFINE_string('ann_path','coco/annotations','set ann_path')
flags.DEFINE_string('restart_param_path',None,'set restart_param_path.')
flags.DEFINE_string('params_path', 'params', 'set params_path.', short_name='p')
flags.DEFINE_string('logs_path', 'logs', 'set logs_path.', short_name='l')
flags.DEFINE_integer('num_worker',4,'set num_worker.')
flags.DEFINE_integer('batch_size',16,'set batch_size.')
flags.DEFINE_string('opt_level','O1','set opt_level.')
flags.DEFINE_bool('augmentation',True,'set augmentation.')
FLAGS = flags.FLAGS

start_lr = 0.001

def train(model, loader, criterion, optimizer):
    '''Traning'''
    model.train()
    total_loss, total_num ,total_intersection, total_union = 0, 0, 0, 0.0
    num,total_time = 0,0
    for feed in tqdm(loader):
        num += 1
        t0 = time.time()
        # prepare Data
        inputs, labels = feed
        labels_np = np.array(labels).astype(int)
        inputs, labels = inputs.cuda(), labels.cuda()
        t1 = time.time()
        # forward & calcurate Loss
        outputs = model(inputs)
        loss = criterion(outputs, labels.type_as(outputs))
        t2 = time.time()
        # initialize gradient
        optimizer.zero_grad()
        # backward
        with amp.scale_loss(loss,optimizer) as scaled_loss:
            scaled_loss.backward()
        # update params
        optimizer.step()
        # totalize score
        t3 = time.time()
        
        pred = np.array(outputs.data.argmax(1).cpu())
        t3_0 = time.time()
        
        for c in range(labels_np.shape[1]):
            pred_area = np.where(pred == c,1,0)
            total_intersection += (pred_area & labels_np[:,c]).sum()
            total_union += (pred_area | labels_np[:,c]).sum()
        t3_1 = time.time()
        total_loss += loss.item() * len(labels_np)
        t3_2 = time.time()
        
        total_num += len(labels_np)
        
        t4 = time.time()
        total_time += t4-t0
        print('\npred:\t{}\nI&U:\t{}\nloss:\t{}\nnum:\t{}'.format(t3_0-t3, t3_1-t3_0, t3_2-t3_1, t4-t3_2))
        if num % 20 == 1 :
            print('\nTOTAL:\t\t{}\nAVERAGE:\t{}\nPER IT:\t\t{}\ntoCUDA:\t\t{}\nFW & LOSS:\t{}\nBW & UD:\t{}\nSCORING:\t{}'.format(total_time,total_time/num,t4-t0,t1-t0,t2-t1,t3-t2,t4-t3))
    return total_loss / total_num, total_intersection / total_union # loss, IoU

def validate(model, loader, criterion):
    '''Validation'''
    model.eval()
    with torch.no_grad():
        total_loss, total_num, total_intersection, total_union = 0, 0, 0, 0.0
        for feed in tqdm(loader):
            # prepare Data
            inputs, labels = feed
            labels_np = np.array(labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            # forward & calcurate Loss
            outputs = model(inputs)
            loss = criterion(outputs, labels.type_as(outputs))
            # totalize score
            pred = np.array(outputs.argmax(1).cpu())
            for c in range(labels_np.shape[1]):
                pred_area = np.where(pred == c,1,0)
                total_intersection += (pred_area & labels_np[:,c]).sum()
                total_union += (pred_area | labels_np[:,c]).sum()
            total_loss += loss.item() * len(labels_np)
            total_num += len(labels)
    return total_loss / total_num, total_intersection / total_union # loss, IoU

def main(argv=None):

    if FLAGS.augmentation:
        transform_train = aug.transform_train
    else:
        transform_train = aug.transform_valid
    train_loader = load_data(
        train_data_path=FLAGS.train_data_path,
        valid_data_path=FLAGS.valid_data_path,
        ann_path=FLAGS.ann_path,
        batch_size=FLAGS.batch_size,
        num_worker=FLAGS.num_worker,
        valid=False,
        transform=transform_train)
    valid_loader = load_data(
        train_data_path=FLAGS.train_data_path,
        valid_data_path=FLAGS.valid_data_path,
        ann_path=FLAGS.ann_path,
        batch_size=FLAGS.batch_size,
        num_worker=FLAGS.num_worker,
        valid=True,
        transform=aug.transform_valid)
    
    model = models.get_model(model_name=FLAGS.model_name)
    if FLAGS.restart_param_path is not None:
        model.load_state_dict(torch.load(FLAGS.restart_param_path))
        if FLAGS.restart_param_path[-9:-4] == 'final':
            start_epoch = int(FLAGS.restart_param_path[-14:-10])
        else:
            start_epoch = int(FLAGS.restart_param_path[-8:-4])+1
    else:
        start_epoch = 0
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level=FLAGS.opt_level)

    criterion = nn.BCELoss().cuda()
    
    LOGS_DIR = os.path.join(FLAGS.logs_path,FLAGS.case,FLAGS.model_name[5:],'tensorboardX')
    writer = SummaryWriter(log_dir=LOGS_DIR)
    PARAMS_DIR = os.path.join(FLAGS.params_path,FLAGS.case,FLAGS.model_name[5:])
    makedirs(PARAMS_DIR,exist_ok=True)
    best_IoU = 0
    total_time=0
    for epoch in range(start_epoch,FLAGS.final_epoch):
        start = time.time()
        lr = cosine_annealing(optimizer, start_lr,epoch,100)
        train_loss, train_IoU = train(model, train_loader, criterion, optimizer)
        valid_loss, valid_IoU = validate(model,valid_loader, criterion)
        print('Epoch: {}, Train Loss: {:.4f}, Train IoU: {:.4f}, Valid Loss: {:.4f}, Valid IoU:{:.4f}'.format(epoch, train_loss, train_IoU, valid_loss, valid_IoU))
        writer.add_scalar('LearningRate', lr, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_IoU', train_IoU, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)
        writer.add_scalar('valid_IoU', valid_IoU, epoch)
        if epoch == FLAGS.final_epoch-1:
            torch.save(model.state_dict(),os.path.join(PARAMS_DIR, '{}_{}_final.pth'.format(FLAGS.case,str(epoch).zfill(4))))
        else:
            torch.save(model.state_dict(),os.path.join(PARAMS_DIR, '{}_{}.pth'.format(FLAGS.case,str(epoch).zfill(4))))
        if valid_IoU > best_IoU:
            best_IoU = valid_IoU
            torch.save(model.state_dict(),os.path.join(PARAMS_DIR,'{}_best.pth'.format(FLAGS.case)))
        total_time += time.time() - start
        print('average_time:{}'.format(total_time/(epoch+1)))

if __name__ == '__main__':
    flags.mark_flags_as_required(['case','model_name'])
    app.run(main)