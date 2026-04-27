#!/usr/bin/env python3
import os
import sys
import json
import base64
import ssl
from pathlib import Path
import subprocess

# Ensure requests and Pillow are installed
try:
    import requests
    from PIL import Image
except ImportError:
    print("Installing missing dependencies (requests, Pillow)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "Pillow"])
        import requests
        from PIL import Image
    except Exception as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

# Disable SSL verification (common issue on macOS python)
ssl._create_default_https_context = ssl._create_unverified_context

def load_env():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        key, value = line.split("=", 1)
                        os.environ[key] = value
                    except ValueError:
                        pass

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_thumbnail(prompt, output_filename, image_paths):
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env or environment")
        sys.exit(1)

    if not image_paths:
        print("Error: No input images provided.")
        sys.exit(1)

    url = "https://api.openai.com/v1/images/edits"
    
    print(f"Generating thumbnail using gpt-image-1.5 on /images/edits (JSON mode)...")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Input Images: {', '.join(image_paths)}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Encode all images to Base64 Data URLs
    images_payload = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Error: Image not found at {path}")
            sys.exit(1)
        base64_data = encode_image(path)
        images_payload.append({ "image_url": f"data:image/png;base64,{base64_data}" })
    
    payload = {
        "model": "gpt-image-1.5",
        "prompt": prompt,
        "images": images_payload,
        "input_fidelity": "low",
        "size": "1536x1024",
        "quality": "low",
        "output_format": "png"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            sys.exit(1)
            
        result = response.json()
        
        # Handle b64_json response
        image_data = None
        
        if 'data' in result and len(result['data']) > 0:
            item = result['data'][0]
            if 'b64_json' in item:
                image_data = base64.b64decode(item['b64_json'])
            elif 'url' in item:
                image_data = requests.get(item['url']).content
                
        if not image_data:
            print(f"No image data found. keys: {result.keys()}")
            sys.exit(1)

        # Save to file (Temporary before crop)
        output_dir = Path("thumbnails")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / output_filename
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        # --- 16:9 AUTO-CROP LOGIC ---
        # 1536x1024 (Original Output) -> 1536x864 (16:9 Target)
        # We need to remove 160px from vertical, so 80px from top and 80px from bottom.
        try:
            print("Applying 16:9 center crop (1536x864)...")
            img = Image.open(output_path)
            width, height = img.size
            
            target_width = 1536
            target_height = 864
            
            # Ensure it's the expected size before cropping
            if width == 1536 and height == 1024:
                top = (height - target_height) // 2
                bottom = top + target_height
                img_cropped = img.crop((0, top, target_width, bottom))
                img_cropped.save(output_path)
                print(f"Thumbnail cropped to 16:9 and saved to: {output_path}")
            else:
                print(f"Skipping crop: Unexpected dimensions {width}x{height}")
                
        except Exception as e:
            print(f"Error during cropping: {e}")
            
        return str(output_path)
        
    except Exception as e:
        print(f"Error generating image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_thumbnail.py <prompt> <output_filename> <image_path_1> [image_path_2] ...")
        sys.exit(1)
        
    prompt = sys.argv[1]
    filename = sys.argv[2]
    image_paths = sys.argv[3:]
    
    generate_thumbnail(prompt, filename, image_paths)
