from typing import Any, Dict

import google.generativeai as genai
from langchain.output_parsers import PydanticOutputParser

from models import AdDetails, AdStoryboardRequest


class AdDetailsGenerator:
    """Generator for ad details using Gemini model with Pydantic output parsing."""

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.7, top_p=0.95):
        """Initialize the generator with model configuration."""
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                temperature=temperature, top_p=top_p
            ),
        )
        self.output_parser = PydanticOutputParser(pydantic_object=AdDetails)

    def generate_ad_details(self, request: AdStoryboardRequest) -> AdDetails:
        """Generate ad details based on the provided request."""
        format_instructions = self.output_parser.get_format_instructions()

        prompt = f"""
        Based on the provided advertisement information, generate comprehensive ad details.
        Fill in any missing information based on the context and marketing best practices.
        
        {format_instructions}
        
        Input information:
        - Product Name: {request.product_name or "Not specified"}
        - Product Description: {request.product_description or "Not specified"}
        - Target Audience: {request.target_audience or "Not specified"}
        - Selling Point: {request.selling_point or "Not specified"}
        - Key Message: {request.key_message or "Not specified"}
        - Duration of Ad: {request.duration_of_ad or "Not specified"}
        - Tone: {request.tone or "Not specified"}
        - Purpose: {request.purpose or "Not specified"}
        - Country: {request.country or "Not specified"}
        - Number of Scenes: {request.number_of_scenes}
        
        Guidelines:
        1. If a product name is not provided, create a unique and memorable name based on the product description.
        2. If product description is not provided, create a comprehensive description of what the product is and its benefits.
        3. If target audience is not specified, determine the most appropriate target audience based on the product.
        4. If selling point is not specified, identify the most compelling unique value proposition.
        5. If key message is not specified, craft a resonant core message aligned with the selling point.
        6. If duration is not specified, suggest an appropriate length based on industry standards.
        7. If tone is not specified, recommend a tone that matches the product and target audience.
        8. If purpose is not specified, determine whether it's for brand awareness, product launch, etc.
        9. If country is not specified, use "International" or "Global".

        For all responses, be specific, creative, and marketing-focused.
        """

        # Generate detailed response
        response = self.model.generate_content(prompt)

        # Parse the response using the Pydantic parser
        try:
            return self.output_parser.parse(response.text)
        except Exception as e:
            # If parsing fails, make a second attempt with a more structured format request
            retry_prompt = f"""
            {prompt}
            
            Your previous response couldn't be parsed correctly. Please ensure you structure your response 
            in a valid JSON format following the exact structure below:
            
            ```json
            {{
                "product_name": "Product Name",
                "product_description": "Comprehensive product description",
                "target_audience": "Specific target demographic",
                "selling_point": "Main unique value proposition",
                "key_message": "Core message for the ad",
                "duration_of_ad": "Duration in seconds/minutes",
                "tone": "Emotional tone of the ad",
                "purpose": "Purpose of the advertisement",
                "country": "Target market",
                "number_of_scenes": {request.number_of_scenes}
            }}
            ```
            
            Provide only this JSON structure in your response, without any additional text.
            """

            retry_response = self.model.generate_content(retry_prompt)
            return self.output_parser.parse(retry_response.text)
