"""Cloud-Based API Engine for IDM-VTON Model Integration.

This module provides the connection interface to call a remote IDM-VTON model hosted
on Hugging Face spaces or Google Colab (exposed via an Ngrok or Gradio tunnel).
"""

import logging
from gradio_client import Client, handle_file

logger = logging.getLogger(__name__)

def call_cloud_api(
    person_img_path: str,
    garment_img_path: str,
    category: str = "upper_body",
    client_url: str = "yisol/IDM-VTON"
):
    """
    Calls the cloud-based IDM-VTON model to process the virtual try-on.

    Args:
        person_img_path: Absolute path to the person image.
        garment_img_path: Absolute path to the garment image.
        category: 'upper_body', 'lower_body', or 'dress'. Default is 'upper_body'.
        client_url: Hugging Face space name or custom Ngrok/Gradio sharing URL.

    Returns:
        A tuple/list with (composite_image_path, mask_image_path) on success.
    """
    logger.info(f"Connecting to Cloud IDM-VTON API at: {client_url}")
    client = Client(client_url)
    
    dict_img = {
        "background": handle_file(person_img_path),
        "layers": [],
        "composite": None
    }
    
    # IDM-VTON category standard: 'upper_body', 'lower_body', or 'dress'
    if category.lower() in ("tshirt", "shirt", "jacket", "t-shirt", "t_shirt", "upper_body"):
        api_category = "upper_body"
    elif category.lower() in ("lowers", "pants", "shorts", "lower_body"):
        api_category = "lower_body"
    else:
        api_category = "upper_body"
        
    logger.info(f"Invoking IDM-VTON tryon prediction with category: {api_category}")
    result = client.predict(
        dict=dict_img,
        garm_img=handle_file(garment_img_path),
        garment_des="A stylish " + api_category,
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )
    logger.info(f"Cloud API result received: {result}")
    return result
