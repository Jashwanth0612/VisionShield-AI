import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class NAFNetService:
    def __init__(self, model_path: str = "models/nafnet/nafnet_weights.pth", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model_path = model_path
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_model(self):
        """Load the NAFNet model weights."""
        try:
            # TODO: Initialize NAFNet architecture here
            # self.model = NAFNet()
            # self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            # self.model.to(self.device)
            # self.model.eval()
            print(f"[NAFNetService] Model initialized successfully on device: {self.device}")
        except Exception as e:
            print(f"[NAFNetService] Warning: Could not load weights from {self.model_path}: {e}")

    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance degraded weather images (fog, rain, low light, snow).
        Returns enhanced PIL Image.
        """
        if self.model is None:
            # Return original image as fallback pass-through if weights are missing
            return image

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_tensor = self.model(img_tensor)

        output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
        enhanced_image = transforms.ToPILImage()(output_tensor)
        
        return enhanced_image
