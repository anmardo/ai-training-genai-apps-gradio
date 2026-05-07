from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# 1. SUMMARIZE
# disclaimer: since i don't have a huggingface key, I translate the function from the course to use openAI api calls

text = ('''The tower is 324 metres (1,063 ft) tall, about the same height
        as an 81-storey building, and the tallest structure in Paris. 
        Its base is square, measuring 125 metres (410 ft) on each side. 
        During its construction, the Eiffel Tower surpassed the Washington 
        Monument to become the tallest man-made structure in the world,
        a title it held for 41 years until the Chrysler Building
        in New York City was finished in 1930. It was the first structure 
        to reach a height of 300 metres. Due to the addition of a broadcasting 
        aerial at the top of the tower in 1957, it is now taller than the 
        Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the 
        Eiffel Tower is the second tallest free-standing structure in France 
        after the Millau Viaduct.''')

from openai import OpenAI
client = OpenAI()

def get_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": f"Summarize this:\n\n{prompt}"
            }
        ]
    )

    return response.choices[0].message.content

import gradio as gr
def summarize(input_text):
    return get_completion(input_text)

gr.close_all()
demo = gr.Interface(
    fn=summarize,
    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
    outputs=[gr.Textbox(label="Result", lines=3)],
    title="Text summarization with gpt-4.1-mini",
    description="Summarize any text using OpenAI models!",
    examples=[text]
)
# demo.launch(share=True, server_port=int(os.environ['PORT1'])) --this doesn't work in corporate laptops!
demo.launch(share=True) # share link cannot be created on gft laptop

# merger tokens function, it only makes sense on the example of the training, not used here
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

gr.close_all()


