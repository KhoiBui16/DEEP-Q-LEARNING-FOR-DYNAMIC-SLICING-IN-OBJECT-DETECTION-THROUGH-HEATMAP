import os


def choose_cam_type():
    type_of_cams = int(input("\nEnter the type of CAM you want to compute (0 - CAM, 1 - Grad-CAM, 2 - Grad-CAM++, 3 - Score-CAM, 4 - Layer-CAM, 5 - XGrad-CAM, 6 - Augmented-GradCAM, 7 - Gridging-gap-WSOL, 8 - F-CAM, 9 - GAN-CAM): "))
    return type_of_cams

def calculate_cam_size_ratio(cam, image, type_of_cams):
    """
    Calculate the ratio of Grad-CAM's size to the original image size.

    Args:
        cam (np.ndarray): The Grad-CAM heatmap (before resizing).
        image (PIL.Image): The original input image.

    Returns:
        float: The ratio of Grad-CAM size to the original image size.
    """
    # Get the original image size
    image_width, image_height = image.size

    # Get the size of the CAM (Grad-CAM before resizing)
    cam_height, cam_width = cam.shape

    # Calculate the area of the image and CAM
    image_area = image_width * image_height
    cam_area = cam_width * cam_height

    # Calculate the ratio of the Grad-CAM area to the image area
    size_ratio = (cam_area / image_area ) * 100

    print(f"\nOriginal image size: {image_width}x{image_height} = {image_area} pixels")
    print(f"{type_of_cams} size: {cam_width}x{cam_height} = {cam_area} pixels")
    print(f"{type_of_cams} to original image size ratio: {size_ratio:.6f}%")
    
    return size_ratio