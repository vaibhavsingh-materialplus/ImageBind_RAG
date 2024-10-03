from vectordb import client, models


def upsert_embeddings(audio_embeded, images_embeded, text_embeded, documents):
    a = client.count(collection_name="imagebind_data")
    client.upload_points(
        collection_name="imagebind_data",
        points=[
            models.PointStruct(
                id=a.count + idx,  # unique id of a point, pre-defined by user
                vector={
                    "audio": audio_embeded[idx],  # embeded audio
                    "image": images_embeded[idx],  # embeded image
                    "text": text_embeded[idx]

                },
                payload=doc  # original image and its caption
            )
            for idx, doc in enumerate(documents)
        ]
    )

    #print("upserting done", "a", a.count)
    print("[INpyFO] Upserting done and count=",a.count)

