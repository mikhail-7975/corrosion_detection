import numpy as np
import cv2
import torch
import albumentations as albu
import segmentation_models_pytorch as smp

def get_validation_augmentation():
    test_transform = [
        albu.Resize(224,224),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


class Model:
    def __init__(self) -> None:
        self.model = torch.load('./unet_corrosion_100epochs.pth')  # load the model
        self.augmentation = get_validation_augmentation()
        self.preprocessing = get_preprocessing(preprocessing_fn)
        self.DEVICE = 'cuda'
    
    def preparing(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        in_image = self.augmentation(image=image)
        in_image = in_image['image']

        in_image = self.preprocessing(image=in_image)
        in_image = in_image['image']

        return image, in_image

    def inference(self, img_path):
        image, in_image = self.preparing(img_path)
        h, w = image.shape[:2]

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(in_image).to(self.DEVICE).unsqueeze(0)
            out = self.model(x_tensor).squeeze().cpu().numpy().round()
            
        out = out.transpose(1,2,0)
        out = cv2.resize(out, (w, h)).astype('uint8')
        out = out[:,:,1]
        
        _, thresh = cv2.threshold(out, 0.5, 255, cv2.THRESH_BINARY_INV)
        pure = np.zeros(thresh.shape).astype('uint8')
        out3 = cv2.merge((pure, pure, thresh))

        res = cv2.addWeighted(image, 1, out3, 0.5, 0)
        
        cv2.imwrite("result.jpg", res)


if __name__ == '__main__':
    model = Model()
    model.preparing('3.jpg')
    model.inference()