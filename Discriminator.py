from quart import Quart, request, render_template
import os
from werkzeug.utils import secure_filename
import zipfile
from FineTune import FineTune
from Classify import Classify

app = Quart(__name__)

# Configure upload settings
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'zip', 'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/finetune', methods=['POST'])
async def finetune_endpoint():
	"""
	Endpoint to handle fine-tuning with uploaded zip file containing image folders
	"""
	if 'file' not in await request.files:
		return {'error': 'No file provided'}, 400

	file = (await request.files)['file']

	if file.filename == '':
		return {'error': 'No selected file'}, 400

	if not allowed_file(file.filename):
		return {'error': 'Invalid file type'}, 400

	if file:
		# Save the uploaded zip file
		filename = secure_filename(file.filename)
		zip_path = os.path.join(UPLOAD_FOLDER, filename)
		await file.save(zip_path)
		
		# Create a directory for extracted contents
		extract_dir = os.path.join(UPLOAD_FOLDER, filename.rsplit('.', 1)[0])
		os.makedirs(extract_dir, exist_ok=True)
		
		# Extract the zip file
		try:
			with zipfile.ZipFile(zip_path, 'r') as zip_ref:
				zip_ref.extractall(extract_dir)
			
			# TODO: Implement actual fine-tuning process
			return {
				'message': 'File uploaded and extracted successfully',
				'extract_path': extract_dir
			}, 200
			
		except Exception as e:
			return {'error': f'Error processing zip file: {str(e)}'}, 500
		
	return {'error': 'Unknown error occurred'}, 500

@app.route('/classify', methods=['POST'])
async def classify_endpoint():
	"""
	Endpoint to handle image classification
	"""
	if 'file' not in await request.files:
		return {'error': 'No file provided'}, 400

	file = (await request.files)['file']

	if file.filename == '':
		return {'error': 'No selected file'}, 400

	if not allowed_file(file.filename):
		return {'error': 'Invalid file type'}, 400

	if file:
		# Save the uploaded image
		filename = secure_filename(file.filename)
		image_path = os.path.join(UPLOAD_FOLDER, filename)
		await file.save(image_path)
		
		try:
			# Use Classify module to process the image
			model_path = "openai/clip-vit-base-patch32"
			classifier = Classify(model_path)
			
			results = classifier.run(image_path)  # Assuming `Classify.run` returns a dictionary
			
			return {
				'message': 'Classification successful',
				'filename': filename,
				'results': results
			}, 200
			
		except Exception as e:
			return {'error': f'Error processing image: {str(e)}'}, 500
		
	return {'error': 'Unknown error occurred'}, 500


@app.route('/', methods=['GET'])
async def index():
	"""
	Simple index page with upload forms
	"""
	return await render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5000)
