import gradio as gr
from PIL import Image
import numpy as np
from vectordb import client as qdrant_client,models
from imagebind import data
import torch
from model import model,device
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import os
from audio_preprocessing import convert_audio
import shutil
from multimodal_embeddings import main,delete_directory

db_dir="database"
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

def singledata(text, audio, image):
    save_dir = "saved_files"
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Saving the audio file in the local
    print("audio", audio)
    # Get the file name from the uploaded audio object
    file_name = str(audio).split("/")[-1]

    # Define the full file path where the file will be saved
    save_path = os.path.join(save_dir, file_name)
    # Now, preprocessing the audio file and then saving it in the "saved_files" directory
    #convert_audio(audio, f"./saved_files/{file_name}")
    convert_audio(audio, save_path)
    save_path = os.path.join(db_dir, file_name)
    convert_audio(audio, save_path)
    # Now, Saving the image to the local using the same code above
    print("vision", image)
    save_path = os.path.join(save_dir, image.split("/")[-1])
    shutil.copy(image, save_path)
    save_path = os.path.join(db_dir, image.split("/")[-1])
    shutil.copy(image, save_path)

    embeds = main(text)
    return f"{embeds}"


def process_text(text_query, audio_query, image_query):
    if text_query and not (audio_query or image_query):
        user_query = [text_query]
        dtype, modality = ModalityType.TEXT, 'title'
        user_input = {dtype: data.load_and_transform_text(user_query, device)}
    elif audio_query and text_query == '' and not (image_query):
        user_query = [audio_query]
        dtype, modality = ModalityType.AUDIO, 'audio'
        user_input = {dtype: data.load_and_transform_audio_data(user_query, device)}
    elif image_query and text_query == '' and not (audio_query):
        user_query = [image_query]
        dtype, modality = ModalityType.VISION, 'image'
        user_input = {dtype: data.load_and_transform_vision_data(user_query, device)}

    with torch.no_grad():
        user_embeddings = model(user_input)

    title_hits = qdrant_client.search(
        collection_name='imagebind_data',
        query_vector=models.NamedVector(
            name="text",
            vector=user_embeddings[dtype][0].tolist(),),
            # search_params=models.SearchParams(hnsw_ef=128, exact=False)

        limit=1  # The limit should be passed here, not inside NamedVector
    )

    audio_hits = qdrant_client.search(
        collection_name='imagebind_data',
        query_vector=models.NamedVector(
            name="audio",
            vector=user_embeddings[dtype][0].tolist(),),
            # search_params=models.SearchParams(hnsw_ef=128, exact=False),
    limit=1,
        )

    image_hits = qdrant_client.search(
        collection_name='imagebind_data',
        query_vector=models.NamedVector(
            name="image",
            vector=user_embeddings[dtype][0].tolist(),),
            # search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=1
        )
    return (title_hits[0].payload['text'], audio_hits[0].payload['audio'], image_hits[0].payload['vision'])


# Gradio Interface
iface1 = gr.Interface(
    title="Combining ImageBind with Qdrant: Vector Similarity Search Across Audio, Video, Text, Image",

    fn=singledata,
    inputs=[
        gr.Textbox(label="text_query"),
        gr.Audio(sources="upload", type="filepath"),
        gr.Image(label="image_query", type="filepath")
    ],
    outputs="text"

)



iface2 = gr.Interface(
    title="Combining ImageBind with Qdrant: Vector Similarity Search Across Audio, Video, Text, Image",

    fn=process_text,
    inputs=[
        gr.Textbox(label="text_query"),
        gr.Audio(sources="upload", type="filepath"),
        gr.Image(label="image_query", type="filepath")
    ],
    outputs=[gr.Textbox(label="Text"),
             gr.Audio(label="Audio"),
             gr.Image(label="Image")],
)

# Add a button for the new function
with iface2:
    btn = gr.Button("Delete Button")
    btn.click(delete_directory("./database"), inputs=[], outputs=gr.Textbox(label="New Function Output"))


demo = gr.TabbedInterface([iface1, iface2],
                          ["Uploading", "Retrieval"])

demo.launch()