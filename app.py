from audiocraft.models import MusicGen
import streamlit as st
import os
import torch
import torchaudio
import base64

@st.cache_resource
def load_model():
    # method inside audiocraft/models/musicgen.py to load musicgen model
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    return model

# function that generates tensors based on the input description and duration
def generate_music_tensors(description, duration: int):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model()

    # set_generation_params function in musicgen.py to set parameters for the model

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    # generates music tensor values eg. Music Tensors:  tensor([[[-0.2991, -0.3089, -0.2977,  ...,  0.0097,  0.0024,  0.0136]]])
    # The audio outputs are a three-dimensional Torch tensor
    # For example, if the tensor is of shape (T, F, D), where T is the number of time steps, F is the number of features, and D is the depth or additional dimension, you might interpret it as follows:

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]

def save_audio(samples: torch.Tensor):
    """Renders an audio player for the given audio samples and saves them to a local directory.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
        save_path (str): path to the directory where audio should be saved.
    """

    print("Samples (inside function): ", samples)

    # rate at which audio will play
    sample_rate = 32000
    save_path = "audio_output/"
    # assertion check to ensure that the samples tensor has either two or three dimensions. If the condition is not met, an AssertionError will be raised.
    assert samples.dim() == 2 or samples.dim() == 3

    # detaches the samples tensor from the computation graph and moves it to the CPU
    samples = samples.detach().cpu()
    # checks if the samples tensor has two dimensions. If it does, it adds a singleton dimension at the beginning using [None, ...]. This is done to ensure that the tensor has a consistent shape before iterating over it.
    if samples.dim() == 2:
        samples = samples[None, ...]

    # Starts a loop that iterates over the `samples` tensor. The loop iterates over each audio sample contained in the `samples` tensor.
    for idx, audio in enumerate(samples):
        # constructs the file path for saving the audio file. It uses the os.path.join() function to concatenate the save_path with a filename composed of "audio_", the loop index idx, and the file extension ".wav".
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        # Saves the audio sample to the specified file path using the `torchaudio.save()` function. It saves the audio sample as a WAV file with the given sample rate.
        torchaudio.save(audio_path, audio, sample_rate)

# function to download the audio file
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon= "musical_note",
    page_title= "Music Gen"
)

def main():
    st.header("Text to Music Generator App🎵")

    with st.expander("About the App"):
        st.write("Music Generator app built using Meta's Audiocraft library's Music Gen Small model. Generates music based on natural language input by the user.")

    text_area = st.text_area("Enter your description:")
    time_slider = st.slider("Select time duration (in seconds)", 2, 20, 5)
    generate_button = st.button("Generate Music")

    if text_area and time_slider and generate_button:
        st.json({
            "Your description": text_area,
            "Selected time duration (in seconds)": time_slider
        })

        st.subheader("Generated Music")

        with st.spinner("Generating..."):

            # passing query and time duration in tensors function
            music_tensors = generate_music_tensors(text_area, time_slider)
            print("Music Tensors: ", music_tensors)
            # saves audio in local directory to render
            save_audio(music_tensors)
            audio_filepath = 'audio_output/audio_0.wav'
            audio_file = open(audio_filepath, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ == '__main__':
    main()