import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from models import AdDetails, ImageToVideoPrompt, Scene, SceneOutput, TextToImagePrompt


class SceneGenerator:
    """Generator for ad scenes using Gemini model with Pydantic output parsing."""

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.7, top_p=0.95):
        """Initialize the generator with model configuration."""
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                temperature=temperature, top_p=top_p
            ),
        )
        self.scene_output_parser = PydanticOutputParser(pydantic_object=SceneOutput)
        self.txt2img_output_parser = PydanticOutputParser(
            pydantic_object=TextToImagePrompt
        )
        self.img2vid_output_parser = PydanticOutputParser(
            pydantic_object=ImageToVideoPrompt
        )

    def generate_scenes(
        self, ad_details: AdDetails, input_scenes: List[Scene] = None
    ) -> List[SceneOutput]:
        """Generate detailed scenes based on ad details and any provided scene inputs."""
        scenes_output = []

        # Determine how many scenes to generate
        if not input_scenes:
            input_scenes = []

        num_scenes = ad_details.number_of_scenes

        for i in range(1, num_scenes + 1):
            # Get input scene if available, otherwise use an empty Scene
            input_scene = input_scenes[i - 1] if i <= len(input_scenes) else Scene()

            # Generate detailed scene
            scene_output = self._generate_single_scene(i, input_scene, ad_details)
            scenes_output.append(scene_output)

        return scenes_output

    def _generate_single_scene(
        self, scene_number: int, input_scene: Scene, ad_details: AdDetails
    ) -> SceneOutput:
        format_instructions = self.scene_output_parser.get_format_instructions()

        # Build priority instructions based on user input
        priority_instructions = ""
        if input_scene.user_recommendation:
            priority_instructions = f"""
        CRITICAL USER SCENE DESCRIPTION - MUST INCLUDE VERBATIM:
        {input_scene.user_recommendation}

        STRICT RULES:
        1. ALL elements below must align with this description
        2. NEVER contradict or omit any details from it
        3. Only add technical details (camera/sound) that support this vision
        4. Preserve all:
        - Character actions 
        - Dialogue 
        - Key visuals 
            """

        prompt = f"""
        Generate a detailed scene for an advertisement storyboard based on the provided information.
        
        
        Ad Details:
        - Product Name: {ad_details.product_name}
        - Product Description: {ad_details.product_description}
        - Target Audience: {ad_details.target_audience}
        - Selling Point: {ad_details.selling_point}
        - Key Message: {ad_details.key_message}
        - Duration of Ad: {ad_details.duration_of_ad}
        - Tone: {ad_details.tone}
        - Purpose: {ad_details.purpose}
        - Country/Target Market: {ad_details.country}
        
        Input Scene Details (Scene {scene_number}):
        - Characters: {input_scene.characters or "Not specified"}
        - Environment: {input_scene.environment or "Not specified"}
        - Time of Day: {input_scene.timing_of_day or "Not specified"}
        - Scene Duration: {input_scene.scene_duration or "Not specified"}
        - Scene Goal: {input_scene.scene_goal or "Not specified"}
        - User Recommendations: {input_scene.user_recommendation or "Not specified (follow normal generation)"}
        
        {priority_instructions}

        this is the format isntructions
        {format_instructions}

        Generation Rules:
        1. IF USER RECOMMENDATIONS EXIST:
        - Implement them EXACTLY as written
        - All other fields must support and align with them
        - Do not add interpretations or modifications
        2. IF NO USER RECOMMENDATIONS:
        - Generate comprehensive scene details following marketing best practices
        - Ensure alignment with ad details and product goals

        Guidelines for scene creation:
        1. If characters are not specified, create appropriate characters for the target audience and the target market.
        2. If environment is not specified, create a suitable setting based on the product and audience.
        3. If time of day is not specified, select an appropriate time that enhances the scene.
        4. If scene duration is not specified, suggest an appropriate duration based on the scene's importance and the duration of the ad.
        5. If scene goal is not specified, determine a goal that supports the overall ad purpose.
        6. Create detailed descriptions for each of these required output components:
           - Visual elements (environment details, character appearances, actions)
           - Camera work (angles, movements, focus points)
           - Sound design (music, effects, dialogue/narration)
           - Transition to the next scene
        7. Each scene should clearly contribute to the overall ad narrative and selling points.
        8. The scene must clearly mention the Key Message: {ad_details.key_message}
        
        Be creative, specific, and focused on marketing effectiveness.
        """

        # Generate detailed scene
        response = self.model.generate_content(prompt)

        # Parse the response
        try:
            return self.scene_output_parser.parse(response.text)
        except Exception as e:
            # If parsing fails, attempt with a more structured format
            retry_prompt = f"""
            {prompt}
            
            Your previous response couldn't be parsed correctly. Please ensure you structure your response 
            in a valid JSON format following the exact structure below:
            
            ```json
            {{
                "scene_number": {scene_number},
                "characters": "Detailed character descriptions",
                "environment": "Detailed environment description",
                "timing_of_day": "Specific time of day",
                "scene_duration": "Duration in seconds",
                "scene_goal": "Specific goal for this scene",
                "visuals": "Detailed visual elements description",
                "camera_work": "Camera angles and movements",
                "sound_design": "Music, effects, and dialogue",
                "transition": "Transition to the next scene"
            }}
            ```
            
            Provide only this JSON structure in your response, without any additional text.
            """

        retry_response = self.model.generate_content(retry_prompt)
        return self.scene_output_parser.parse(retry_response.text)

    def generate_text_to_image_prompts(
        self, scenes: List[SceneOutput]
    ) -> List[TextToImagePrompt]:
        """Generate text-to-image prompts for all scenes."""
        txt2img_prompts = []

        for scene in scenes:
            # Generate text-to-image prompt for the scene
            txt2img_prompt = self._generate_single_txt2img_prompt(scene)
            txt2img_prompts.append(txt2img_prompt)

        return txt2img_prompts

    def _generate_single_txt2img_prompt(self, scene: SceneOutput) -> TextToImagePrompt:
        format_instructions = self.txt2img_output_parser.get_format_instructions()

        # Check for user recommendations in the original scene if available
        user_rec = getattr(scene, "user_recommendation", None)

        priority_instructions = ""
        if user_rec:
            priority_instructions = f"""
            USER VISUAL INSTRUCTIONS (MUST FOLLOW EXACTLY):
            {user_rec}

            RULES:
            1. These visual descriptions take PRECEDENCE over all generated content
            2. Translate them DIRECTLY into the image prompt without modification
            3. Ensure all prompt sections align with these instructions
            """

        prompt = f"""

        Create a detailed text-to-image prompt for Scene {scene.scene_number} following this EXACT format:
        {format_instructions}

        TEXT-TO-IMAGE PROMPT FOR SCENE {scene.scene_number}:
        CHARACTERS:
        [For each character in the scene]
        - Character Name: [Character Name]
        - Appearance: [age, gender, distinctive physical features]
        - Clothing: [detailed outfit description + accessories]
        - Expression: [specific emotion + facial details]

        ACTION: 
        - [Precise single-moment action using active verbs]
        - *Example:* "Right hand gripping sword hilt while left arm shields face"

        ENVIRONMENT:
        - Location: [specific setting + key objects]
        - Lighting: [time of day + light quality]
        - Details: [notable atmospheric elements]

        STYLE:
        - Type: Photorealistic CGI(MUST USE THIS EXACTLY)
        - Technique: [specific rendering technique like "tilt-shift depth of field"]
        - Reference:Gregory Crewdson (MUST USE THIS EXACTLY)


        ---
        

        
        Scene Details to Incorporate:
        - Characters: {scene.characters}
        - Environment: {scene.environment}
        - Time of Day: {scene.timing_of_day}
        - Visuals: {scene.visuals}
        - Camera Work: {scene.camera_work}
        
        {priority_instructions}

        STRICT RULES:
        1. NEVER merge categories - keep CHARACTERS/ACTION/ENVIRONMENT/STYLE strictly separated
        2. The prompt must describe a SINGLE frozen moment (no sequential actions)
        3. Prioritize atomic details (e.g. "frayed red shoelace on left boot")
        4. If user instructions exist, implement them exactly as provided
        5. Use active voice and specific visual descriptors
        6. For scenes with multiple characters, include ALL characters with separate Appearance, Clothing, and Expression fields
        7. Format each character's details on separate lines as shown in the CHARACTER section
        """

        # Generate text-to-image prompt
        response = self.model.generate_content(prompt)

        # Parse the response
        try:
            txt2img_prompt = self.txt2img_output_parser.parse(response.text)
            # Ensure scene number is correct
            txt2img_prompt.scene_number = scene.scene_number
            return txt2img_prompt
        except Exception as e:
            # If parsing fails, attempt with a more structured format
            retry_prompt = f"""
                {prompt}
                
                Your previous response couldn't be parsed correctly. Please ensure you structure your response 
                in a valid JSON format following the exact structure below:
                
                ```json
                {{
                    "scene_number": {scene.scene_number},
                    "characters": [
                        {{
                            "name": "Character Name",
                            "appearance": "Detailed appearance description",
                            "clothing": "Detailed clothing description",
                            "expression": "Specific facial expression"
                        }}
                    ],
                    "action": "Precise single-moment action",
                    "environment_location": "Specific setting description",
                    "environment_lighting": "Detailed lighting description",
                    "environment_details": "Notable environmental elements",
                    "style_type": "Photorealistic CGI",
                    "style_technique": "Specific rendering technique",
                    "style_reference": "Gregory Crewdson"
                }}
            Provide only this JSON structure in your response, without any additional text.
            """

        retry_response = self.model.generate_content(retry_prompt)
        txt2img_prompt = self.txt2img_output_parser.parse(retry_response.text)
        txt2img_prompt.scene_number = scene.scene_number
        return txt2img_prompt

    def generate_image_to_video_prompts(
        self, scenes: List[SceneOutput], txt2img_prompts: List[TextToImagePrompt]
    ) -> List[ImageToVideoPrompt]:
        """Generate image-to-video prompts for all scenes using both scene details and text-to-image prompts."""
        img2vid_prompts = []

        for scene, txt2img_prompt in zip(scenes, txt2img_prompts):
            # Generate image-to-video prompt for the scene
            img2vid_prompt = self._generate_single_img2vid_prompt(scene, txt2img_prompt)
            img2vid_prompts.append(img2vid_prompt)

        return img2vid_prompts

    def _generate_single_img2vid_prompt(
        self, scene: SceneOutput, txt2img_prompt: TextToImagePrompt
    ) -> ImageToVideoPrompt:
        format_instructions = self.img2vid_output_parser.get_format_instructions()

        # Check for user recommendations in the original scene if available
        user_rec = getattr(scene, "user_recommendation", None)

        priority_instructions = ""
        if user_rec:
            priority_instructions = f"""
            USER VISUAL INSTRUCTIONS (MUST FOLLOW EXACTLY):
            {user_rec}
            
            RULES:
            1. These visual descriptions take PRECEDENCE over all generated content
            2. Translate them DIRECTLY into the video prompt without modification
            3. Ensure all prompt sections align with these instructions
            """

        prompt = f"""
        Create a detailed image-to-video prompt for Scene {scene.scene_number} following this EXACT format:
        {format_instructions}
        IMAGE-TO-VIDEO PROMPT FOR SCENE {scene.scene_number}:
        CHARACTER:
`       [For each character in the scene]
        - Character Name: [Character Name]
        - Appearance: [age, gender, distinctive physical features]
        - Clothing: [detailed outfit description + accessories]
        - Expression: [specific emotion + facial details]`

        ACTION: 
        - Timed Action: [precise 0.5s-3s action with timing markers]
        - Object Interaction: [specific items being manipulated]
        - Motion FX: [required effects like "smear frames" or "impact blur"]

        ENVIRONMENT:
        - FG: [Interactive foreground elements with parallax details]
        - MG: [Midground setting + specific camera movement]
        - BG: [Atmospheric background + depth effects]
        - Lighting: [dynamic lighting changes if any]

        STYLE:
        - Type: Photorealistic CGI (MUST USE THIS EXACTLY)
        - Technique: [specific rendering technique like "motion blur + smear frames"]
        - Reference: Gregory Crewdson (MUST USE THIS EXACTLY)
        - FPS: [required frame rate, typically 24 or 30]
        - Camera: [specific camera movement and lens details]
        - Negative Prompts: [elements to avoid in generation]

        ---
        
        
        
        BASE SCENE DETAILS:
        - Characters: {scene.characters}
        - Environment: {scene.environment}
        - Time of Day: {scene.timing_of_day}
        - Visuals: {scene.visuals}
        - Camera Work: {scene.camera_work}
        
        TEXT-TO-IMAGE PROMPT TO ANIMATE:
        - Characters: {', '.join([f'Character Name: {char.name}, Appearance: {char.appearance}, Clothing: {char.clothing}, Expression: {char.expression}' for char in txt2img_prompt.characters])}
        - Action: {txt2img_prompt.action}
        - Environment: {txt2img_prompt.environment_location}, {txt2img_prompt.environment_lighting}
        - Style: {txt2img_prompt.style_type}, {txt2img_prompt.style_technique}
        
        {priority_instructions}
        WAN2.1 STRICT RULES:
        1. Time ALL actions (e.g. "1.2s sword draw from scabbard")
        2. Specify hair/cloth physics (e.g. "left pigtail swings 30° right on impact")
        3. Use anime motion terms: "smear", "impact frame", "stretch frame"
        4. Camera MUST move (no static shots) - specify movement type
        5. Include parallax and layered depth in environment
        6. Use exact durations for all actions and transitions
        7. Each prompt must represent a SINGLE high-impact motion sequence
        8. Specify at least 3 keyframes in the action description
        9. For scenes with multiple characters, include ALL characters with separate Appearance, Clothing Physics, Expression, and Base Pose fields
        10. Format each character's details on separate lines as shown in the CHARACTER section
        """

        # Generate image-to-video prompt
        response = self.model.generate_content(prompt)

        # Parse the response
        try:
            img2vid_prompt = self.img2vid_output_parser.parse(response.text)
            # Ensure scene number is correct
            img2vid_prompt.scene_number = scene.scene_number
            return img2vid_prompt
        except Exception as e:
            # If parsing fails, attempt with a more structured format
            retry_prompt = f"""
            {prompt}
        
            Your previous response couldn't be parsed correctly. Please ensure you structure your response 
            in a valid JSON format following the exact structure below:
            
            ```json
            {{
                "scene_number": {scene.scene_number},
                "characters": [
                    {{
                        "name": "Character Name",
                        "appearance": "1girl/1boy, age, body type with hair motion details",
                        "clothing": "Clothing with physics description",
                        "expression": "Expression with timing details",
                        "base_pose": "Starting pose before action"
                    }}
                ],
                "action": "Timed action with object interaction and motion FX",
                "environment_location": "FG/MG/BG with parallax details",
                "environment_lighting": "Dynamic lighting description",
                "environment_details": "Layered environment specifics",
                "style_type": "Photorealistic CGI",
                "style_technique": "Specific animation technique",
                "style_reference": "Gregory Crewdson"
            }}
    Provide only this JSON structure in your response, without any additional text.
    """

        retry_response = self.model.generate_content(retry_prompt)
        img2vid_prompt = self.img2vid_output_parser.parse(retry_response.text)
        img2vid_prompt.scene_number = scene.scene_number
        return img2vid_prompt
