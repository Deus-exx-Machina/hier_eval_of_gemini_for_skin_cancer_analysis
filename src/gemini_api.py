import logging
import base64
import time

from io import BytesIO
from PIL import Image

from google import genai
from google.genai.types import GenerateContentConfig

LOGGER = logging.getLogger(__name__)


class GeminiAPIHandler:
    def __init__(self, api_key, index, model_name="gemini-2.0-flash-exp",
                 temperature=0.0, topP=0.95, topK=1, seed=42,
                 request_interval=6, logger=None):
        """
        Initialize the Gemini API Client.

        Parameters:
        - api_key: str - Your Gemini API key.
        - model_name: str - The model to use (default: "gemini-2.0-flash-exp").
        - max_requests_per_minute: int - Maximum requests allowed per minute (default: 10).
        - logger: logging.Logger - Optional logger for debug information.
        - worker_id: str - Identifier for the worker (default: "unknown").
        """
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.index = index
        self.model = model_name
        self.interval = request_interval
        self.logger = LOGGER if logger is None else logger
        self.config = GenerateContentConfig(
            temperature=temperature,
            topP=topP,
            topK=topK,
            seed=seed
        )

    def generate_text(self, prompt):
        """
        Test method to generate text content from a prompt.

        Parameters:
        - prompt: str - The text prompt to send to the model.

        Returns:
        - str - The generated text.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.config
            )
            return response.text
        
        except Exception as e:
            print(f"Error generating text: {e}")
            return None


    def generate_from_image_path(self, image_path, prompt):
        """
        Generate content based on an image and a text prompt.

        Parameters:
        - image_path (str): Path to the image file.
        - prompt (str): The text prompt to guide the generation.

        Returns:
        - str: The generated response text.
        """
        image = Image.open(image_path)
        return self._generate_with_retry(image, prompt)


    def generate_from_pil_image(self, pil_image, prompt):
        """
        Generate content based on a single PIL Image object and a text prompt.

        Parameters:
        - pil_image (PIL.Image.Image): A PIL Image object.
        - prompt (str): The text prompt to guide the generation.

        Returns:
        - str: The generated response text.
        """
        # Ensure the image is in RGB mode (convert from RGBA if necessary)
        if pil_image.mode == "RGBA":
            LOGGER.warning(
                "Image is in RGBA mode, converting to RGB for JPEG compatibility."
            )
            pil_image = pil_image.convert("RGB")
        elif pil_image.mode != "RGB":
            LOGGER.warning(
                f"Image is in {pil_image.mode} mode, converting to RGB for compatibility."
            )
            pil_image = pil_image.convert("RGB")

        # Keep the .jpeg format in memory and retrieve it as a byte string for faster processing
        with BytesIO() as img_buffer:
            pil_image.save(img_buffer, format="JPEG")
            image_bytes = img_buffer.getvalue()
        image = Image.open(BytesIO(image_bytes))

        return self._generate_with_retry(image, prompt)
        
    
    def _generate_with_retry(self, image, prompt):
        """
        Attempt generation with one retry on 429 RESOURCE_EXHAUSTED error.

        Parameters:
        - image: PIL Image object
        - prompt: str

        Returns:
        - str: The generated response text
        """
        try:
            return self._attempt_generate(image, prompt)
        except Exception as e:
            is_quota_exceeded = False # Check if it's a 429 RESOURCE_EXHAUSTED type error

            # Check common signatures of the error
            if hasattr(e, 'args') and len(e.args) > 0:
                msg = str(e.args[0])
                if 'RESOURCE_EXHAUSTED' in msg or '429' in msg or 'quota' in msg.lower():
                    is_quota_exceeded = True

            if is_quota_exceeded:
                self.logger.warning(f"[Handler_{self.index}] Quota exceeded. Attempting retry.")

                retry_delay = 30  # default fallback delay
                try:
                    # Try to parse retry delay from exception message if available
                    if 'RetryInfo' in msg:
                        import re
                        match = re.search(r'retryDelay.*?(\d+(\.\d+)?)s', msg)
                        if match:
                            retry_delay = int(float(match.group(1)))
                except Exception:
                    self.logger.warning(f"[Handler_{self.index}] Failed to parse retry delay; using default 30s.")

                self.logger.warning(f"[Handler_{self.index}] Sleeping for {retry_delay}s before retry.")
                time.sleep(retry_delay)

                try:
                    return self._attempt_generate(image, prompt)
                except Exception as retry_e:
                    self.logger.error(f"[Handler_{self.index}] Retry failed: {retry_e}")
                    raise retry_e

            # Not a 429 error — re-raise the original exception
            raise e
        
    
    def _attempt_generate(self, image, prompt):
        """
        Internal helper to call the API.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=[image, prompt],
            config=self.config
        )
        time.sleep(self.interval)
        return response.text


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=f"../.env")

    gemini = GeminiAPIHandler(api_key=os.getenv("GEMINI_API_KEY"), index=0)

    text_prompt = "Do you know da wae?"
    print("Text Response:", gemini.generate_text(text_prompt))
