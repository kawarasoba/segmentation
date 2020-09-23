import torch
import numpy as np
import matplotlib.pyplot as plt
from absl import flags, app
from albumentations import Compose

from dataset import load_data,CocoDataset
import models
import augmentations as aug

flags.DEFINE_string('case', None, 'set case.', short_name='c')
flags.DEFINE_string('data_path', 'coco', 'set train_images_path.', short_name='ti')
flags.DEFINE_string('params_path', None, 'set params_path.', short_name='p')
FLAGS = flags.FLAGS


def visualize(image, ground_truth_mask, predicted_mask):
    '''plot images in one row.'''
    image = image.transpose(1,2,0)
    print('image.shape:{}'.format(image.shape))
    transform = Compose([aug.Normalize(mean=(-1,-1,-1),std=(2,2,2),max_pixel_value=1.0)])
    image = transform(image=image)['image']
    print('(aug)image -> max:{}~min:{}'.format(np.max(image),np.min(image)))
    ground_truth = np.zeros((image.shape[0],image.shape[1]))
    predicted = np.zeros((image.shape[0],image.shape[1]))
    category = []   
    for c in range(len(ground_truth_mask)):
        ground_truth += ground_truth_mask[c]*(1/255/80.0*c)
        if(ground_truth_mask[c].sum() != 0):
            predicted += predicted_mask[c]*(1/255/80.0*c)
            category.append(c)
    plt.figure(figsize=(16,5))
    images = {
        'name':['image','ground_truth','predicted'],
        'item':[image,ground_truth,predicted]}
    print('category:{}'.format(category))
    for i in range(len(images['item'])):
        plt.subplot(1,len(images['item']),i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title('{}'.format(images['name'][i]))
        plt.imshow(images['item'][i])
    plt.show()

def inference(model, loader,device):
    model.eval()
    with torch.no_grad():
        total_intersection, total_union = 0, 0.0
        # prepare Data
        inputs, labels = next(loader.__iter__())
        inputs, labels = inputs.to(device), labels.to(device)
        # forward
        print('inputs.shape:{},labels.shape:{}'.format(inputs.shape,labels.shape))
       
        outputs = model(inputs).cpu()
        predicted_mask = np.zeros_like(np.array(outputs)).astype(np.int8)
        # totalize score
        pred = np.array(outputs.argmax(1,keepdim=True))
        labels = np.array(labels.cpu())
        for c in range(labels.shape[1]):
            pred_area = np.where(pred[:,0] == c,1,0)
            predicted_mask[:,c] += pred_area            
            total_intersection += (pred_area & labels[:,c]).sum()
            total_union += (pred_area | labels[:,c]).sum()
        #visualize
        print('IoU:{}'.format(total_intersection/total_union))
        visualize(
            image=np.array(inputs.cpu()).squeeze(),
            ground_truth_mask=labels.squeeze(),
            predicted_mask=predicted_mask.squeeze())
    return total_intersection / total_union

def main(argv=None):

    loader = load_data(
        batch_size=1,
        data_path=FLAGS.data_path,
        valid=True,
        transform=aug.transform_inference)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = models.get_model(model_name='Unet_'+FLAGS.params_path.split('/')[2])
    model.to(device=device)
    model.load_state_dict(torch.load(FLAGS.params_path))

    IoU = inference(model,loader,device)
    print('IoU:{}'.format(IoU))

if __name__ == '__main__':
    flags.mark_flags_as_required(['case','params_path'])
    app.run(main)