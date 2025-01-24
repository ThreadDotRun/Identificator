from quart import Quart, request, render_template, session
import os
import secrets
from werkzeug.utils import secure_filename
import zipfile
from FineTune import FineTune
from Classify import Classify

app = Quart(__name__)

# Configuration setup
ALLOWED_EXTENSIONS = {'zip', 'png', 'jpg', 'jpeg'}
BASE_UPLOAD_PATH = './images'

# Secure secret key setup
if (secret_key := os.environ.get("QUART_SECRET_KEY")) is None:
	print("WARNING: QUART_SECRET_KEY not set. Generating a temporary key.")
	secret_key = secrets.token_hex(32)
app.secret_key = secret_key

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_upload_folder(filename):
	"""Dynamically generate upload path based on session ID"""
	user = "root"  # Hardcoded user from original implementation
	session_id = session.get('_id', 'default_session')
	return os.path.join(BASE_UPLOAD_PATH, f"{user}_{session_id}_{filename}")

@app.route('/finetune', methods=['POST'])
async def finetune_endpoint():
	upload_folder = get_upload_folder()
	os.makedirs(upload_folder, exist_ok=True)

	if 'file' not in await request.files:
		return {'error': 'No file provided'}, 400

	file = (await request.files)['file']

	if file.filename == '':
		return {'error': 'No selected file'}, 400

	if not allowed_file(file.filename):
		return {'error': 'Invalid file type'}, 400

	try:
		filename = secure_filename(file.filename)
		zip_path = os.path.join(upload_folder, filename)
		await file.save(zip_path)
		
		extract_dir = os.path.join(upload_folder, filename.rsplit('.', 1)[0])
		os.makedirs(extract_dir, exist_ok=True)
		
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(extract_dir)
			
		return {
			'message': 'File uploaded and extracted successfully',
			'extract_path': extract_dir
		}, 200
		
	except Exception as e:
		return {'error': f'Error processing zip file: {str(e)}'}, 500

from quart import request
import asyncio

@app.route('/classify', methods=['POST'])
async def classify_endpoint():
    try:
        # Get form data asynchronously
        form_data = await request.form
        files = await request.files
        
        # Validate file
        if 'file' not in files:
            return {'error': 'No file provided'}, 400
            
        file = files['file']
        classes = form_data.get('classes', '')
        
        # Validate inputs
        if not file.filename:
            return {'error': 'No selected file'}, 400
        if not classes:
            return {'error': 'No classes provided'}, 400

        # Process class list
        class_list = [cls.strip() for cls in classes.split(',') if cls.strip()]
        if not class_list:
            return {'error': 'Empty class list'}, 400
        if len(class_list) > 5:
            return {'error': 'Maximum 5 classes allowed'}, 400

        # File handling
        if not allowed_file(file.filename):
            return {'error': 'Invalid file type'}, 400

        filename = secure_filename(file.filename)
        upload_folder = get_upload_folder(filename)
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, filename)
        
        # Save file asynchronously
        await file.save(image_path)
        
        # Run synchronous classifier in thread pool
        model_path = "openai/clip-vit-base-patch32"
        classifier = Classify(model_path)
        
        # Wrap synchronous call in async thread executor
        predicted_class, confidence = await asyncio.to_thread(
            classifier.predict,  # Your synchronous method
            image_path,
            class_list
        )
        
        return {
            'message': 'Classification successful',
            'filename': filename,
            'results': {predicted_class: confidence}
        }, 200
        
    except Exception as e:
        return {'error': f'Error processing image: {str(e)}'}, 500
        
    except Exception as e:
        return {'error': f'Error processing image: {str(e)}'}, 500
        

@app.route('/', methods=['GET'])
async def index():
	return await render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5000)