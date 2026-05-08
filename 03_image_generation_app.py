from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import os


# 1. GENERATE IMAGES, again I am using openAI equivalent since I have a key for it
# Helper function
from openai import OpenAI
from IPython.display import display, Image
import base64

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Text-to-image endpoint
def get_completion(prompt, width, height):
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size= f"{width}x{height}"
    )

    return response.data[0].b64_json


prompt = "a dog in the park"
#writeImage(get_completion(prompt, 1024, 1024), "output")

def writeImage(result, image_name) :
    pil_image = PILImage.fromarray(result.astype("uint8"))
    pil_image.save(f"{image_name}.png")

    #    f.write(base64.b64decode(result))
    print("displayed!")


# 2. GRADIO interface

import gradio as gr
from PIL import Image as PILImage
import io

gr.close_all()

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = PILImage.open(byte_stream)
    return pil_image

def generate(prompt, width, height):
    output = get_completion(prompt, width, height)
    result_image = base64_to_pil(output)
    return result_image

#demo = gr.Interface(
#    fn=generate,
#    inputs=[
#        gr.Textbox(label="Your prompt"),
#        gr.Slider(label="Width (Supported sizes in OpenAI are 1024x1024, 1024x1536, 1536x1024)", minimum=1024, maximum=1536, step=512, value=1024),
#        gr.Slider(label="Height (Supported sizes in OpenAI are 1024x1024, 1024x1536, 1536x1024)", minimum=1024, maximum=1536, step=512, value=1024)
#    ],
#    outputs=gr.Image(label="Result"),
#    title="Image Generation with OpenAI",
#    description="Generate any image with OpenAI"
#)
#demo.launch()

# 2. GRADIO blocks

with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with OpenAI")
    prompt = gr.Textbox(label="Your prompt")
    with gr.Accordion("Height and Width", open=True):
        with gr.Row():
            with gr.Column():
                width = gr.Slider(label="Width (Supported sizes in OpenAI are 1024x1024, 1024x1536, 1536x1024)", minimum=1024, maximum=1536, step=512, value=1024)
                height = gr.Slider(label="Height (Supported sizes in OpenAI are 1024x1024, 1024x1536, 1536x1024)", minimum=1024, maximum=1536, step=512, value=1024)
    with gr.Row():
        with gr.Column():
            btn = gr.Button("Submit")
    with gr.Row():
        with gr.Column():
            output = gr.Image(label="Result")
            downloadBtn = gr.Button("Download")
            filename = gr.Textbox(value="image", visible=False)

    btn.click(fn=generate, inputs=[prompt,width,height], outputs=[output])
    downloadBtn.click(fn=writeImage, inputs=[output,filename], outputs=[])

gr.close_all()
demo.launch()
