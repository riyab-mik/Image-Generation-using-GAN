from flask import Flask, render_template, jsonify, request
import torch
from model import Generator
from torchvision.transforms import ToPILImage, Normalize
import io
import base64

app = Flask(__name__)

# Load the pre-trained GAN generator model
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(latent_dim).to(device)

# Load state_dict with strict=False to handle mismatches
try:
    state_dict = torch.load("generator_final.pth", map_location=device)
    generator.load_state_dict(state_dict, strict=False)
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    raise  # Re-raise to ensure the application doesn't continue with a broken model

generator.eval()

# Custom normalization - Adjust this based on how your GAN model was trained
normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        # Create latent vector z
        z = torch.randn(1, latent_dim, device=device)
        
        with torch.no_grad():
            generated_image = generator(z).detach().cpu().squeeze(0)

        # If the generator outputs values between [-1, 1], denormalize them
        generated_image = normalize(generated_image)

        # Convert to PIL image
        to_pil = ToPILImage()
        generated_image = to_pil(generated_image)
        
        # Optionally, resize the image for better display quality
        generated_image = generated_image.resize((1024, 1024))  # Increase resolution for quality
        
        # Save image to a BytesIO object
        buffer = io.BytesIO()
        generated_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert image to base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({'success': True, 'image': image_base64})
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)