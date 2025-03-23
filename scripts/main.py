from torchvision import transforms
from PIL import Image
import os
from cam_utils import CAMProcessor
from grad_cam_utils import GradCAMProcessor
from grad_cam_pp_utils import GradCAMPlusPlusProcessor
from score_cam_utils import ScoreCAMProcessor
from layer_cam_utils import LayerCAMProcessor
from xgrad_cam_utils import XGradCAMProcessor
from augmented_grad_cam_utils import AugmentedGradCAMProcessor
from utils import choose_cam_type, calculate_cam_size_ratio
from config import type_cams, class_idx, transform, script_dir, image_path, WEIGHTS, MODEL, TARGET_LAYER



if __name__ == '__main__':
    image = Image.open(image_path)
    image_name = os.path.basename(image_path)
    img_tensor = transform(image).unsqueeze(0)

    # Load a pre-trained model
    weights = WEIGHTS
    model = MODEL
    target_layer = TARGET_LAYER # target_layer to create heatmap
    
    model.eval()
    
    # Choose the type of CAM
    type_of_cams = choose_cam_type()
    
    heatmap_save_dir = os.path.join(script_dir, f"../heatmaps/{type_cams[type_of_cams]}")
    if not os.path.exists(heatmap_save_dir):
        os.makedirs(heatmap_save_dir)
    
    if type_of_cams == 0:
        print("Computing CAM...")
    
        # Initialize CAMProcessor
        cam_processor = CAMProcessor(model, target_layer)

        # Compute CAM
        # cam = cam_processor.compute_custom_cam(img_tensor)
        cam = cam_processor.compute_custom_cam_by_torchcam(img_tensor, class_idx)
        
        # Overlay CAM on the image
        overlayed_image = cam_processor.overlay_cam_on_image(image, cam)

        # Display and save the overlayed image
        save_path = os.path.join(heatmap_save_dir, f'CAM_heatmap_{image_name}')
        cam_processor.display_and_save_overlay_cam(image, overlayed_image, save_path)
        
        # Calculate the size ratio of the CAM
        calculate_cam_size_ratio(cam, image, "CAM")

    elif type_of_cams == 1:
        print("Computing Grad-CAM...")
    
        # Initialize Grad-CAM
        grad_cam_processor = GradCAMProcessor(model, target_layer)
        
        # Compute Grad-CAM
        # grad_cam = grad_cam_processor.compute_grad_cam(img_tensor)
        grad_cam = grad_cam_processor.compute_grad_cam_by_torchcam(img_tensor, class_idx)
        
        # Overlay Grad-CAM on the image
        overlayed_image = grad_cam_processor.overlay_cam_on_image(image, grad_cam)
        
        # Display and save the overlayed image
        save_path = os.path.join(heatmap_save_dir, f'GradCAM_heatmap_{image_name}')
        grad_cam_processor.display_and_save_overlay_cam(image, overlayed_image, save_path)
        
        # Calculate the size ratio of the Grad-CAM
        calculate_cam_size_ratio(grad_cam, image, "Grad-CAM")

    
    elif type_of_cams == 2:
        print("Computing Grad-CAM++...")
    
        # Initialize Grad-CAM++
        grad_cam_pp_processor = GradCAMPlusPlusProcessor(model, target_layer)
        
        # Compute Grad-CAM++
        # grad_cam_pp = grad_cam_pp_processor.compute_grad_cam_plus_plus(img_tensor)
        grad_cam_pp = grad_cam_pp_processor.compute_grad_cam_plus_plus_by_torchcam(img_tensor, class_idx)
        
        # Overlay Grad-CAM++ on the image
        overlayed_image = grad_cam_pp_processor.overlay_cam_on_image(image, grad_cam_pp)
        
        # Display and save the overlayed image
        save_path = os.path.join(heatmap_save_dir, f'GradCAMpp_heatmap_{image_name}')
        grad_cam_pp_processor.display_and_save_overlay_cam(image, overlayed_image, save_path)
        
        # Calculate the size ratio of the Grad-CAM++
        calculate_cam_size_ratio(grad_cam_pp, image, "Grad-CAM++")
        
    elif type_of_cams == 3:
        print("Computing Score-CAM...")
        
        score_cam_processor = ScoreCAMProcessor(model, target_layer)
        
        # Compute Score-CAM
        score_cam = score_cam_processor.compute_score_cam_by_torchcam(img_tensor, class_idx)
        
        # Overlay Score-CAM on the image
        overlayed_image = score_cam_processor.overlay_cam_on_image(image, score_cam)
        
        # Display and save the overlayed image
        save_path = os.path.join(heatmap_save_dir, f'ScoreCAM_heatmap_{image_name}')
        score_cam_processor.display_and_save_overlay_cam(image, overlayed_image, save_path)
        
        # Calculate the size ratio of the Score-CAM
        calculate_cam_size_ratio(score_cam, image, "Score-CAM")
    
    elif type_of_cams == 4:
        print("Computing Layer-CAM...")
        layer_cam_processor = LayerCAMProcessor(model, target_layer)
        
        # Compute Layer-CAM
        layer_cam = layer_cam_processor.compute_layer_cam_by_torchcam(img_tensor, class_idx)
        
        # Overlay Layer-CAM on the image
        overlayed_image = layer_cam_processor.overlay_cam_on_image(image, layer_cam)
        
        # Display and save the overlayed image
        save_path = os.path.join(heatmap_save_dir, f'LayerCAM_heatmap_{image_name}')
        layer_cam_processor.display_and_save_overlay_cam(image, overlayed_image, save_path)
        
        # Calculate the size ratio of the Layer-CAM
        calculate_cam_size_ratio(layer_cam, image, "Layer-CAM")
    
    elif type_of_cams == 5:
        print("Computing XGrad-CAM...")
        xgrad_cam_processor = XGradCAMProcessor(model, target_layer)
        
        # Compute XGrad-CAM
        xgrad_cam = xgrad_cam_processor.compute_xgrad_cam_by_torchcam(img_tensor, class_idx)
        
        # Overlay XGrad-CAM on the image
        overlayed_image = xgrad_cam_processor.overlay_cam_on_image(image, xgrad_cam)
        
        # Display and save the overlayed image
        save_path = os.path.join(heatmap_save_dir, f'XGradCAM_heatmap_{image_name}')
        xgrad_cam_processor.display_and_save_overlay_cam(image, overlayed_image, save_path)
        
        # Calculate the size ratio of the XGrad-CAM
        calculate_cam_size_ratio(xgrad_cam, image, "XGrad-CAM")
    
    elif type_of_cams == 6:
        print("Computing Augmented Grad-CAM...")
        augmented_grad_cam_processor = AugmentedGradCAMProcessor(model, target_layer)
        
        # Compute Augmented Grad-CAM
        augmented_grad_cam = augmented_grad_cam_processor.compute_augmented_grad_cam(img_tensor, class_idx)
        
        # Overlay Augmented Grad-CAM on the image
        overlayed_image = augmented_grad_cam_processor.overlay_cam_on_image(image, augmented_grad_cam)
        
        # Display and save the overlayed image
        save_path = os.path.join(heatmap_save_dir, f'AugmentedGradCAM_heatmap_{image_name}')
        augmented_grad_cam_processor.display_and_save_overlay_cam(image, overlayed_image, save_path)
        
        # Calculate the size ratio of the Augmented Grad-CAM
        calculate_cam_size_ratio(augmented_grad_cam, image, "Augmented Grad-CAM")    
    
    
    else:
        pass