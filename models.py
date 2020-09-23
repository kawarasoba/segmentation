import torch.nn as nn
import segmentation_models_pytorch as smp
import constants as cons

def Unet(encoder_name='',classes=1):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet',
        classes=classes,
        activation='softmax')
    #model.cuda()
    return model

def get_model(model_name='',num_classes=cons.NUM_CLASSES):

    if model_name.startswith('Unet_'):
        if model_name[5:]=='resnet34':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        elif model_name[5:]=='resnet50':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        elif model_name[5:]=='resnext101_32x8d':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        elif model_name[5:]=='senet154':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        elif model_name[5:]=='se_resnet50':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        elif model_name[5:]=='se_resnext50_32x4d':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        elif model_name[5:]=='densenet161':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        elif model_name[5:]=='inceptionv4':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        elif model_name[5:]=='efficientnetb3':
            return Unet(encoder_name=model_name[5:],classes=num_classes)
        else:
            raise ValueError('given Non-exist encoder_name.')
    else:
        raise ValueError('given Non-exist model_name.')
