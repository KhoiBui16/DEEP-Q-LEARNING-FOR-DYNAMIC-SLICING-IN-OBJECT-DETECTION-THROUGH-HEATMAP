import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import ScoreCAM
import cv2


class ScoreCAMProcessor:
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Initialize the Score-CAM Processor.

        Args:
            model (torch.nn.Module): The model to use for Score-CAM.
            target_layer (str): The name of the target layer for CAM.
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.cam_extractor = ScoreCAM(self.model, target_layer=self.target_layer)

    def compute_score_cam_by_torchcam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate the Score-CAM heatmap for a specific class.

        Args:
            input_tensor (torch.Tensor): The input tensor (1, C, H, W).
            class_idx (int): The target class index.

        Returns:
            np.ndarray: The Score-CAM heatmap.
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()
                
            cam = self.cam_extractor(class_idx, output)
            
            # print(f"CAM type {type(cam)}")
            # print(f"CAm is: {cam}")
            if type(cam) == list:
                cam = cam[0] # to extract the first CAM from the list to tensor -  matrix heatmap [1, 7, 7]
            
            # print(cam.shape)
            cam = cam.squeeze(0).cpu().numpy()
            cam -= cam.min()
            cam /= cam.max() if cam.max() != 0 else 1
            
        # print(f"Score-CAM size: {cam.shape}")
        # print(f"Score-CAM resolution: {cam.shape[0]}x{cam.shape[1]} = {cam.shape[0] * cam.shape[1]} pixels")
        
        return cam

    def overlay_cam_on_image(self, image, cam, alpha = 0.7, red_threshold = 0.8, yellow_threshold = 0.5):
        """
        Overlay the Score-CAM heatmap onto the original image, showing only the important regions (from red to yellow) and
        keeping the background intact (showing original image in blue regions).

        Args:
            image (PIL.Image): The original input image.
            cam (np.ndarray): The Score-CAM heatmap.
            alpha (float): The transparency factor for the original image overlaying.
            red_threshold (float): The threshold for red areas. Regions with values greater than this will show heatmap in red.
            yellow_threshold (float): The threshold for yellow areas. Regions with values between red and yellow will show heatmap in yellow.

        Returns:
            PIL.Image: Image with overlayed Score-CAM.
        """
        # Convert PIL.Image to numpy array and RGB to BGR
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Resize heatmap to match original image dimensions
        heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))

        # Normalize heatmap to [0, 255]
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / heatmap.max() if heatmap.max() != 0 else heatmap
        heatmap = np.uint8(255 * heatmap)

        # Apply colormap (JET) to heatmap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Create masks for red, yellow and the remaining regions
        red_mask = heatmap[:, :, 0] > (red_threshold * 255)  # Mask for red regions
        yellow_mask = (heatmap[:, :, 0] > (yellow_threshold * 255)) & (heatmap[:, :, 0] <= (red_threshold * 255))  # Mask for yellow regions
        remaining_mask = ~red_mask & ~yellow_mask  # Mask for the remaining areas (less important regions)

        # Apply heatmap to important regions (red and yellow) with `alpha`
        img[red_mask] = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)[red_mask]  # Apply red regions with `alpha`
        img[yellow_mask] = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)[yellow_mask]  # Apply yellow regions with `alpha`

        # Keep the background intact (remaining areas) with `1 - alpha`
        img[remaining_mask] = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)[remaining_mask]  # Apply background with `1 - alpha`

        # Convert back from BGR to RGB and numpy array to PIL.Image
        overlayed_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return overlayed_image


    def display_and_save_overlay_cam(self, image: Image.Image, overlayed_image, save_path):
        """
        Plot the original image and the Score-CAM heatmap.

        Args:
            image (Image.Image): The original input image.
            overlayed_image (PIL.Image): Image with overlayed Score-CAM.
        """

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(overlayed_image)
        axes[1].set_title("Score-CAM")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        
        # Save the overlayed image
        overlayed_image.save(save_path)
        print(f"Overlayed image saved to {save_path}")


