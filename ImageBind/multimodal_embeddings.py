from model import model,device
import os
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from upserting import upsert_embeddings
import shutil


def delete_directory(directory_path):
    """Delete a directory and all its contents."""
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' has been deleted.")
    else:
        print(f"Directory '{directory_path}' does not exist or is not a directory.")
    return "success"

documents = []  # for metadata

image_extension_set = set(["jpg", "png","jpeg"])


def main(text):
    audio_paths = []
    image_paths = []
    text_list = []

    audio_db_paths = []
    image_db_paths = []
    for file in os.listdir("./saved_files"):
        if str(file).split(".")[-1] == "wav":
            audio_paths.append("./saved_files/" + file)
            audio_db_paths.append("./database/" + file)
        if str(file).split(".")[-1] in image_extension_set:
            image_paths.append("./saved_files/" + file)
            image_db_paths.append("./database/" + file)

    if text=="":
        text_list.append("")
    else:
        text_list.append(str(text))

    documents = [{"audio": audio, "vision": image, "text": str(text)} for audio, image, text in
                 zip(audio_db_paths, image_db_paths, text_list)]
    print(text)
    # Load data
    inputs = {ModalityType.TEXT: data.load_and_transform_text(text_list, device),
              ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
              ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)}
    print("1")

    with torch.no_grad():
        embeddings = model(inputs)

    images_embeded = list(embeddings['vision'])
    audio_embeded = list(embeddings['audio'])
    text_embeded = list(embeddings['text'])

    upsert_embeddings(audio_embeded, images_embeded, text_embeded, documents)

    # Example usage
    delete_directory('./saved_files')
    return embeddings


