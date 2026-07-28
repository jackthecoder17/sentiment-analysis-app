import gradio as gr

def test_batch(texts):
    print("Received:", repr(texts))
    # Return a list of strings, just like the sentiment app
    return ["Original: " + t for t in texts]

with gr.Blocks() as iface:
    input_text = gr.Textbox()
    output_text = gr.Textbox()
    btn = gr.Button("Submit")
    btn.click(test_batch, inputs=input_text, outputs=output_text, batch=True, max_batch_size=8)

iface.launch(prevent_thread_lock=True)
