from io import BytesIO

import google.generativeai as genai
import streamlit as st

from ad_details_generator import AdDetailsGenerator
from fluxstreamlit import generate_images_from_prompts

# Import local modules
from models import (
    AdDetails,
    AdStoryboardRequest,
    ImageToVideoPrompt,
    Scene,
    SceneOutput,
    StoryboardOutput,
    TextToImagePrompt,
)
from scene_generator import SceneGenerator

# from PIL import Image


# Load environment variables from .env
# load_dotenv()

# Configure genai with the API key from .env
# genai.configure(api_key=os.getenv("GENAI_API_KEY"))
genai.configure(api_key=st.secrets["GENAI_API_KEY"])


class AdStoryboardGenerator:
    """Generator for complete advertisement storyboards."""

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.7, top_p=0.95):
        """Initialize the generator with model configuration."""
        self.ad_details_generator = AdDetailsGenerator(model_name, temperature, top_p)
        self.scene_generator = SceneGenerator(model_name, temperature, top_p)

    def generate(self, request: AdStoryboardRequest) -> StoryboardOutput:
        """Generate a complete storyboard based on the provided request."""
        # Generate ad details
        ad_details = self.ad_details_generator.generate_ad_details(request)

        # Generate scenes
        scenes = self.scene_generator.generate_scenes(ad_details, request.scenes)

        # Generate text-to-image prompts
        txt2img_prompts = self.scene_generator.generate_text_to_image_prompts(scenes)

        # Generate image-to-video prompts
        img2vid_prompts = self.scene_generator.generate_image_to_video_prompts(
            scenes, txt2img_prompts
        )

        # Combine all outputs
        storyboard_output = StoryboardOutput(
            ad_details=ad_details,
            scenes=scenes,
            txt2img_prompts=txt2img_prompts,
            img2vid_prompts=img2vid_prompts,
        )

        return storyboard_output

    def format_output(self, storyboard: StoryboardOutput) -> str:
        """Format the storyboard output into a readable string."""
        # Format ad details
        output = "# Advertisement Storyboard\n\n"
        output += "## Product Details\n"
        output += f"- **Name:** {storyboard.ad_details.product_name}\n"
        output += f"- **Description:** {storyboard.ad_details.product_description}\n"
        output += f"- **Target Audience:** {storyboard.ad_details.target_audience}\n"
        output += f"- **Main Selling Point:** {storyboard.ad_details.selling_point}\n"
        output += f"- **Core/Key Message:** {storyboard.ad_details.key_message}\n"
        output += f"- **Total Duration:** {storyboard.ad_details.duration_of_ad}\n"
        output += f"- **Tone:** {storyboard.ad_details.tone}\n"
        output += f"- **Market:** {storyboard.ad_details.country}\n"
        output += f"- **Purpose:** {storyboard.ad_details.purpose}\n\n"
        output += f"- **Total Scenes:** {storyboard.ad_details.number_of_scenes}\n"

        # Format scenes
        output += "## Scene Breakdown\n\n"
        for scene in storyboard.scenes:
            output += f"### Scene {scene.scene_number}\n"
            output += f"- **Characters:** {scene.characters}\n"
            output += f"- **Environment:** {scene.environment}\n"
            output += f"- **Time of Day:** {scene.timing_of_day}\n"
            output += f"- **Duration:** {scene.scene_duration}\n"
            output += f"- **Scene Goal:** {scene.scene_goal}\n\n"
            output += f"**Visuals:**\n{scene.visuals}\n\n"
            output += f"**Camera Work:**\n{scene.camera_work}\n\n"
            output += f"**Sound Design:**\n{scene.sound_design}\n\n"
            output += f"**Transition:**\n{scene.transition}\n\n"

        # Format text-to-image prompts
        imgPrompt = "## Text-to-Image Prompts\n\n"
        for prompt in storyboard.txt2img_prompts:
            imgPrompt += f"### Text-to-Image Prompt for Scene {prompt.scene_number}\n\n"

            # Character section
            imgPrompt += "**CHARACTER:**\n"
            for char in prompt.characters:
                imgPrompt += f"- Character Name: {char.name}: \n"
                imgPrompt += f"- Appearance: {char.appearance}\n"
                imgPrompt += f"- Clothing:  {char.clothing}\n"
                imgPrompt += f"- Expression: {char.expression}\n"
            imgPrompt += "\n"

            # Action section
            imgPrompt += "**ACTION:**\n"
            imgPrompt += f"- {prompt.action}\n\n"

            # Environment section
            imgPrompt += "**ENVIRONMENT:**\n"
            imgPrompt += f"- Location: {prompt.environment_location}\n"
            imgPrompt += f"- Lighting: {prompt.environment_lighting}\n"
            imgPrompt += f"- Details: {prompt.environment_details}\n\n"

            # Style section
            imgPrompt += "**STYLE:**\n"
            imgPrompt += f"- Type: {prompt.style_type}\n"
            imgPrompt += f"- Technique: {prompt.style_technique}\n"
            if prompt.style_reference:
                imgPrompt += f"- Reference: {prompt.style_reference}\n"
            imgPrompt += "\n"

        # Format image-to-video prompts
        vidPrompt = "## Image-to-Video Prompts\n\n"
        for prompt in storyboard.img2vid_prompts:
            vidPrompt += (
                f"### Image-to-Video Prompt for Scene {prompt.scene_number}\n\n"
            )

            # Character section
            vidPrompt += "**CHARACTER:**\n"
            for char in prompt.characters:
                vidPrompt += f"- Character Name: {char.name}: \n"
                vidPrompt += f"- Appearance:  {char.appearance}\n"
                vidPrompt += f"- Clothing: {char.clothing}\n"
                vidPrompt += f"- Expression: {char.expression}\n"
            vidPrompt += "\n"

            # Action section
            vidPrompt += "**ACTION:**\n"
            vidPrompt += f"- {prompt.action}\n\n"

            # Environment section
            vidPrompt += "**ENVIRONMENT:**\n"
            vidPrompt += f"- Location: {prompt.environment_location}\n"
            vidPrompt += f"- Lighting: {prompt.environment_lighting}\n"
            vidPrompt += f"- Details: {prompt.environment_details}\n\n"

            # Style section
            vidPrompt += "**STYLE:**\n"
            vidPrompt += f"- Type: {prompt.style_type}\n"
            vidPrompt += f"- Technique: {prompt.style_technique}\n"
            if prompt.style_reference:
                vidPrompt += f"- Reference: {prompt.style_reference}\n"
            vidPrompt += "\n"

        return output, imgPrompt, vidPrompt


# Gradio interface
def create_gradio_interface():
    st.title("Advertisement Storyboard Generator")
    st.markdown("Create professional ad storyboards with AI assistance")

    # Main product information
    with st.expander("Product Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            product_name = st.text_input(
                "Product Name", help="Name of the product being advertised"
            )
            product_description = st.text_area(
                "Product Description",
                help="Description of product features and benefits",
            )
            target_audience = st.text_input(
                "Target Audience", help="Target demographic for the advertisement"
            )

        with col2:
            selling_point = st.text_input(
                "Main Selling Point", help="Unique value proposition"
            )
            key_message = st.text_input("Key Message", help="Core message to convey")

    # Ad details
    with st.expander("Advertisement Details"):
        col1, col2 = st.columns(2)
        with col1:
            duration_of_ad = st.text_input(
                "Duration of Ad (seconds)", help="Total duration of the advertisement"
            )
            tone = st.text_input(
                "Tone",
                help="Emotional tone or mood (e.g., Inspirational, Funny, Serious, Emotional)",
            )

        with col2:
            purpose = st.text_input(
                "Purpose",
                help="Purpose of the advertisement (e.g., Brand Awareness, Product Launch)",
            )
            country = st.text_input(
                "Target Country/Market", help="Where will this ad run?"
            )

    # Dynamic scene creation
    st.subheader("Storyboard Scenes")
    number_of_scenes = st.number_input(
        "Number of Scenes", min_value=1, max_value=8, value=1, step=1
    )

    scenes = []
    for i in range(number_of_scenes):
        with st.expander(f"Scene {i+1}", expanded=i == 0):
            col1, col2 = st.columns(2)
            with col1:
                characters = st.text_area(
                    f"Characters (Scene {i+1})",
                    key=f"characters_{i}",
                    help="Who appears in this scene",
                )
                environment = st.text_area(
                    f"Environment (Scene {i+1})",
                    key=f"environment_{i}",
                    help="Setting or location",
                )
                user_recommendation = st.text_area(
                    f"User Recommendation (Scene {i+1})",
                    key=f"user_recommendation_{i}",
                    help="any recommendations for actions or so on",
                )

            with col2:
                timing_of_day = st.text_input(
                    f"Time of Day (Scene {i+1})",
                    key=f"timing_{i}",
                    help="Morning, afternoon, night, etc.",
                )
                scene_duration = st.text_input(
                    f"Duration (seconds) (Scene {i+1})",
                    key=f"duration_{i}",
                    help="How long this scene lasts",
                )
                scene_goal = st.text_area(
                    f"Scene Goal (Scene {i+1})",
                    key=f"goal_{i}",
                    help="What this scene accomplishes",
                )

            scenes.append(
                Scene(
                    characters=characters,
                    environment=environment,
                    timing_of_day=timing_of_day,
                    scene_duration=scene_duration,
                    scene_goal=scene_goal,
                    user_recommendation=user_recommendation,
                )
            )

    if st.button("Generate Storyboard"):
        if not product_name:
            st.warning("Please provide at least a product name")
            return

        # Create proper Scene objects
        scenes = []
        for i in range(number_of_scenes):
            scenes.append(
                Scene(
                    characters=st.session_state[f"characters_{i}"],
                    environment=st.session_state[f"environment_{i}"],
                    timing_of_day=st.session_state[f"timing_{i}"],
                    scene_duration=st.session_state[f"duration_{i}"],
                    scene_goal=st.session_state[f"goal_{i}"],
                )
            )

        # Create proper AdStoryboardRequest object
        request = AdStoryboardRequest(
            product_name=product_name,
            product_description=product_description,
            target_audience=target_audience,
            selling_point=selling_point,
            key_message=key_message,
            duration_of_ad=duration_of_ad,
            tone=tone,
            purpose=purpose,
            country=country,
            number_of_scenes=number_of_scenes,
            scenes=scenes,
        )

        # Generate the storyboard
        with st.spinner("Generating your storyboard..."):
            try:
                generator = AdStoryboardGenerator()
                storyboard_output = AdStoryboardGenerator().generate(request)
                formatted_output, imagePrompt, vidPrompt = generator.format_output(
                    storyboard_output
                )
                st.subheader("Generated Storyboard")
                st.markdown(formatted_output)
                # generate_images_from_prompts(imagePrompt)
                # generate_video_from_image(vidPrompt)

                generated_images = generate_images_from_prompts(imagePrompt)

                # Display all generated images with download buttons
                for title, image, download_button in generated_images:
                    st.subheader(title)
                    st.image(image, use_container_width=True)
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

                # print(imagePrompt)
                # print(vidPrompt)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# Run the Gradio interface if this script is executed directly
if __name__ == "__main__":
    create_gradio_interface()
