from typing import List, Optional

from pydantic import BaseModel, Field


class Scene(BaseModel):
    """Represents a single scene in an advertisement storyboard."""

    characters: Optional[str] = Field(
        default=None,
        description=(
            "Characters who appears in the scene: include age, profession, and traits relevant to the ad. "
            "If not specified, choose characters that represent the product's target audience "
            "(e.g., athletes and gym-goers for supplements). Infer details like sport, appearance, "
            "and clothing based on the product and audience."
        ),
    )

    environment: Optional[str] = Field(
        default=None,
        description=(
            "Setting or location, aligned with the target user's habits or product context. "
            "If not specified, choose an environment that reflects the lifestyle of the target audience "
            "or highlights product use (e.g., a kitchen for a cooking gadget). Include visual and sensory details."
        ),
    )

    timing_of_day: Optional[str] = Field(
        default=None,
        description=(
            "Time of day when the scene takes place, adding emotional and contextual depth. "
            "If not specified, choose a time of day that complements the scene's mood and matches realistic usage "
            "(e.g., morning commute, evening jog)."
        ),
    )

    scene_duration: Optional[str] = Field(
        default=None,
        description=(
            "Duration of the scene in seconds; proportionate to total ad length. "
            "If not specified, estimate based on the total ad duration "
            "(e.g., shorter scenes for intro, longer for climax)."
        ),
    )

    scene_goal: Optional[str] = Field(
        default=None,
        description=(
            "Narrative or emotional goal of the scene. "
            "If not specified, determine based on the ad's overall purpose "
            "(e.g., demonstrate product use, evoke curiosity, build excitement)."
        ),
    )
    user_recommendation: Optional[str] = Field(
        default=None,
        description=(
            "STRICT USER INSTRUCTIONS that MUST be followed exactly as provided. "
            "When present, these recommendations take absolute priority over all other fields. "
            "The system must adapt all other scene elements to perfectly match these instructions. "
            "User may specify: VISUALS, CAMERA WORK, SOUND, TRANSITIONS, TIMING, or SPECIAL REQUESTS."
        ),
    )


class AdStoryboardRequest(BaseModel):
    """Input parameters for generating an advertisement storyboard."""

    product_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of the product being advertised. If not provided, give a unique name to the product "
            "based on its description. For example, if the product description is 'noise-cancelling headphones', "
            "give the product a unique name like 'QuietBeats'. If the product is a campaign to emphasize the dangers of drugs, "
            "give a name like 'Addictionless'."
        ),
    )

    product_description: Optional[str] = Field(
        default=None,
        description=(
            "Description of the product features and benefits. If not provided, Describe the product and what would make this product beneficial to the target audience "
            "and what it does."
        ),
    )

    target_audience: Optional[str] = Field(
        default=None,
        description=(
            "Target demographic for the advertisement. If not provided, Determine the target audience according to the product name, description, "
            "and who will benefit from this product. For example, if the ad is about beard oil, then the target audience is males with beards. "
            "Add more detailed descriptions when appropriate."
        ),
    )

    selling_point: Optional[str] = Field(
        default=None,
        description=(
            "Main selling point or unique value proposition. If not provided, Determine the main selling point from the product name and description. "
            "For example, if the product is noise-cancelling headphones, then the main selling point is noise cancellation (not necessarily sound quality)."
        ),
    )

    key_message: Optional[str] = Field(
        default=None,
        description=(
            "Core message to be conveyed in the advertisement. If not provided, Give a unique key message for the product according to the product name, "
            "description, target audience, and country. For example, if the product is noise-cancelling headphones, the key message would be something like "
            "'Isolate yourself from the world' or 'Like you live alone in this world'."
        ),
    )

    duration_of_ad: Optional[str] = Field(
        default=None,
        description=(
            "Total duration of the advertisement. If not provided, Determine the duration from the number of scenes and the duration of each scene.and if this also is not specified determine the ad length based on the norm for that product ads"
        ),
    )

    tone: Optional[str] = Field(
        default=None,
        description=(
            "Emotional tone or mood of the advertisement. If not provided, Determine the best tone for the ad from the product name, product description, "
            "target audience, and country."
        ),
    )

    purpose: Optional[str] = Field(
        default=None,
        description=(
            "Purpose of the advertisement (e.g., brand awareness, product launch). If not provided, Determine the purpose from the product name and product description. "
            "For example, if it's a product to sell, then the purpose is advertisement. If the product is a campaign to stop the abuse of drugs, then the purpose is informative."
        ),
    )

    country: Optional[str] = Field(
        default=None, description="Target country or market for the advertisement."
    )

    number_of_scenes: Optional[int] = Field(
        # default=None,
        description=(
            # "Number of scenes. If not provided, Determine the number of scenes from the total duration of the ad, the number of scenes provided, "
            # "and the duration of each scene. If not specified, estimate to best serve the ad structure."
            "Number of scenes"
        ),
    )

    scenes: Optional[List[Scene]] = Field(
        # default=None,
        description=(
            # "List of scenes in the storyboard. If the number of scenes is less than the expected count to complete a professional ad, "
            # "generate extra scenes accordingly."
            "List of scenes in the storyboard"
        ),
    )


class AdDetails(BaseModel):
    """Output format for ad details generation."""

    product_name: str
    product_description: str
    target_audience: str
    selling_point: str
    key_message: str
    duration_of_ad: str
    tone: str
    purpose: str
    country: str
    number_of_scenes: int


class SceneOutput(BaseModel):
    """Output format for scene generation."""

    scene_number: int
    characters: str
    environment: str
    timing_of_day: str
    scene_duration: str
    scene_goal: str
    visuals: str
    camera_work: str
    sound_design: str
    transition: str


# Define a new class to represent individual character details
class CharacterDetails(BaseModel):
    name: str = Field(description="Name of the character")
    appearance: str = Field(
        description="Physical appearance of the character (e.g., age, skin tone, height)"
    )
    clothing: str = Field(description="Clothing worn by the character")
    expression: str = Field(
        description="Facial expression or emotional state of the character"
    )


class TextToImagePrompt(BaseModel):
    """Output format for text-to-image prompts."""

    scene_number: int
    characters: List[CharacterDetails] = Field(
        description="List of characters with their appearance, clothing, and expression"
    )
    action: str
    environment_location: str
    environment_lighting: str
    environment_details: str
    style_type: str = "Photorealistic CGI"
    style_technique: str
    style_reference: str = "Gregory Crewdson"


class ImageToVideoPrompt(BaseModel):
    """Output format for image-to-video prompts with temporal controls."""

    scene_number: int
    characters: List[CharacterDetails] = Field(
        description="List of characters with their appearance, clothing, and expression"
    )
    action: str
    environment_location: str
    environment_lighting: str
    environment_details: str
    style_type: str = "Photorealistic CGI"
    style_technique: str
    style_reference: str = "Gregory Crewdson"


class StoryboardOutput(BaseModel):
    """Complete storyboard output format combining all parts."""

    ad_details: AdDetails
    scenes: List[SceneOutput]
    txt2img_prompts: List[TextToImagePrompt]
    img2vid_prompts: List[ImageToVideoPrompt]
