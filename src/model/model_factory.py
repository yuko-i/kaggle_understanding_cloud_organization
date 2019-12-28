from model.Linknet_resnet import Linknet_resnet
from model.Linknet_resnet_ASPP import Linknet_resnet_ASPP
from model.Unet_resnet import Unet_resnet


def make_Linknet_model(encoder: str,
                       decoder: str,
                       class_num: int):

    if encoder.startswith('resnet') and decoder == '':
        return Linknet_resnet(encoder=encoder, class_num=class_num)

    elif encoder.startswith('resnet') and decoder == 'ASPP':
        return Linknet_resnet_ASPP(encoder=encoder, class_num=class_num)


def make_Unet_model(encoder: str,
                    decoder: str,
                    class_num: int):

    if encoder.startswith('resnet') and decoder == '':
        return Unet_resnet(encoder=encoder, class_num=class_num)




def make_model(
        model_name: str='Unet',
        encoder: str='resnet18',
        decoder: str='',
        class_num: int=4):

    if model_name == 'Linknet':
        return make_Linknet_model(encoder=encoder,
                                  decoder=decoder,
                                  class_num=class_num)
    if model_name == 'Unet':
        return make_Unet_model(encoder=encoder,
                                  decoder=decoder,
                                  class_num=class_num)
    #if model_name == 'FPN':
    #    return make_FPNnet_model()

    else:
        return make_Linknet_model()
