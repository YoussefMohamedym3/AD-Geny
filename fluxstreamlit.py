import base64
import re
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from replicate import Client

# Load environment variables
# load_dotenv()
# REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
if not REPLICATE_API_TOKEN:
    raise EnvironmentError("REPLICATE_API_TOKEN not found in .env file")

# Initialize Replicate client
client = Client(api_token=REPLICATE_API_TOKEN)


def process_input_text(text):
    split_content = text.split("## Text-to-Image Prompts", 1)
    if len(split_content) > 1:
        return split_content[1].strip()
    return text.strip()


def extract_scenes(text):
    scenes = re.split(r"### Text-to-Image Prompt for Scene \d+", text)
    return [scene.strip() for scene in scenes if scene.strip()]


def format_scene_prompt(scene):
    return re.sub(r"\*\*(.*?)\*\*:", r"\1:", scene).replace("\n", " ")


def get_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def create_download_button(image, filename, button_text):
    """Create a download button for the image"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{button_text}</a>'
    return href


def generate_images_from_prompts(prompt_text):
    """
    Generate images from prompts and return them with download capabilities.

    Args:
        prompt_text (str): The input text containing the prompts

    Returns:
        list: Tuples containing (title, PIL Image, download_button_html)
    """
    generated_images = []

    try:
        processed_text = process_input_text(prompt_text)
        scenes = extract_scenes(processed_text)

        for idx, scene in enumerate(scenes, start=1):
            formatted_prompt = format_scene_prompt(scene)
            st.write(f"Generating image for Scene {idx}...")

            with st.spinner(f"Generating Scene {idx}..."):
                output = client.run(
                    "black-forest-labs/flux-dev",
                    input={
                        "prompt": formatted_prompt,
                        "guidance_scale": 7.5,
                        "num_inference_steps": 30,
                        "seed": 42,
                    },
                )

                if isinstance(output, list):
                    image_url = output[0]
                elif hasattr(output, "url"):
                    image_url = output.url
                else:
                    image_url = output

                image = get_image_from_url(image_url)
                filename = f"scene_{idx}.png"
                download_button = create_download_button(
                    image, filename, f"Download Scene {idx}"
                )
                generated_images.append((f"Scene {idx}", image, download_button))

            st.success(f"Scene {idx} generated!")

        return generated_images

    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise


# Streamlit UI
def main():
    st.title("Image Generation from Prompts")

    # Text input area
    prompt_text = st.text_area("Enter your prompt text:", height=300)

    if st.button("Generate Images"):
        if not prompt_text.strip():
            st.warning("Please enter some prompt text")
        else:
            try:
                generated_images = generate_images_from_prompts(prompt_text)

                # Display all generated images with download buttons
                for title, image, download_button in generated_images:
                    st.subheader(title)
                    st.image(image, use_column_width=True)
                    st.markdown(download_button, unsafe_allow_html=True)

                # Add a "Download All" button
                if len(generated_images) > 1:
                    st.markdown("---")
                    all_images = BytesIO()
                    image.save(
                        all_images,
                        format="PNG",
                        save_all=True,
                        append_images=[img for _, img, _ in generated_images[1:]],
                    )
                    all_images.seek(0)
                    # st.download_button(
                    #     label="Download All Images as ZIP",
                    #     data=all_images,
                    #     file_name="all_scenes.zip",
                    #     mime="application/zip",
                    # )

            except Exception as e:
                st.error(f"Failed to generate images: {e}")


if __name__ == "__main__":
    main()
