from quart import Quart, request, render_template, session, jsonify
import os
import secrets
from werkzeug.utils import secure_filename
import zipfile
from FineTune import FineTune
from Classify import Classify
import asyncio
import traceback
from DatabaseManager import DatabaseManager

# Database connection details
DB_USER = "ident_tdr"
DB_PASSWORD = "8g76yfg87hs8gf7hfs8hgf"
DB_HOST = "127.0.0.1"
DB_NAME = "Identificator_db"
db = DatabaseManager(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)

class_list_length = 10
max_classes_allowed = 10

# Initialize DatabaseManager
db_manager = DatabaseManager(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)

app = Quart(__name__)

# Configuration setup
ALLOWED_EXTENSIONS = {'zip', 'png', 'jpg', 'jpeg'}
BASE_UPLOAD_PATH = './images'
user_max_file_str_len = 1024  # Later - Get from current user
user_max_class_str_len = 64   # Later - Get from current user
MAX_FILES = 10

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
	upload_folder = get_upload_folder("finetune")  # Ensure a unique folder name
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
		
		# Fine-tuning process
		fine_tune = FineTune("openai/clip-vit-base-patch32")
		loss_history = fine_tune.fine_tune(extract_dir)  # Pass the extracted folder to fine_tune method

		# Return loss history for display
		return {
			'message': 'Model fine-tuned successfully',
			'lossHistory': loss_history  # Return the loss history
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

		# Validate file and classes presence
		if 'file' not in files:
			return {'error': 'No file provided'}, 400
		if 'classes' not in form_data:
			return {'error': 'No classes provided'}, 400

		file = files['file']
		classes_str = form_data['classes']

		# Validate file size using content_length
		print(f"File Content Length {file.content_length}")
		if file.content_length > user_max_file_str_len:
			return {'error': f'File size exceeds your limit of {user_max_file_str_len} bytes'}, 400

		# Validate class string length
		if len(classes_str) > user_max_class_str_len:
			return {'error': f'Class list exceeds your limit of {user_max_class_str_len} characters'}, 400

		# Process class list
		class_list = [cls.strip() for cls in classes_str.split(',') if cls.strip()]
		if not class_list:
			return {'error': 'Empty class list'}, 400
		if len(class_list) > max_classes_allowed:
			return {f'error': 'Maximum {max_classes_allowed} classes allowed'}, 400

		# File handling
		if not allowed_file(file.filename):
			return {'error': 'Invalid file type'}, 400

		filename = secure_filename(file.filename)
		upload_folder = get_upload_folder(filename)
		os.makedirs(upload_folder, exist_ok=True)
		image_path = os.path.join(upload_folder, filename)

		# Save file asynchronously
		await file.save(image_path)

		# Run classification
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
		# Capture the full traceback
		error_trace = traceback.format_exc()
		print(f"Error Traceback: {error_trace}")  # Log the traceback for debugging
		return jsonify({'error': f'Error processing image: {str(e)}', 'traceback': error_trace}), 500

# Helper function to check allowed file types
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to get the upload folder path
def get_upload_folder(folder_name):
	base_upload_path = "./uploads"
	return os.path.join(base_upload_path, folder_name)

# Asynchronous route for batch classification
@app.route('/classify/batch', methods=['POST'])
async def classify_batch_endpoint():
	try:
		# Correctly await form and files
		form_data = await request.form
		files = (await request.files).getlist('files')

		# Validate required fields
		if not files:
			return jsonify({'error': 'No files provided'}), 400
		if not form_data.get('classes'):
			return jsonify({'error': 'No classes provided'}), 400

		# Validate the number of files
		num_files = len(files)
		if num_files > MAX_FILES:
			return jsonify({'error': f'Maximum {MAX_FILES} files allowed, but {num_files} provided'}), 400

		# Validate class list string
		classes_str = form_data['classes']
		if len(classes_str) > user_max_class_str_len:
			return jsonify({'error': f'Class list exceeds {user_max_class_str_len} characters'}), 400

		# Parse and validate class list
		class_list = [cls.strip() for cls in classes_str.split(',') if cls.strip()]
		if not class_list:
			return jsonify({'error': 'Empty class list'}), 400
		if len(class_list) > class_list_length:
			return jsonify({'error': 'Maximum 5 classes allowed'}), 400

		# Process and validate files
		upload_folder = get_upload_folder("batch_classify")
		os.makedirs(upload_folder, exist_ok=True)
		image_paths = []

		for file in files:
			if not allowed_file(file.filename):
				return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
			if file.filename == '':
				return jsonify({'error': 'One or more files has no filename'}), 400

			# Save file
			filename = secure_filename(file.filename)
			image_path = os.path.join(upload_folder, filename)
			await file.save(image_path)  # Use Quart's async save
			image_paths.append(image_path)

		# Run batch classification
		model_path = "openai/clip-vit-base-patch32"
		classifier = Classify(model_path)

		# Wrap synchronous batch prediction in async thread
		results = await asyncio.to_thread(
			classifier.predict_batch,
			image_paths,
			class_list
		)

		# Format results
		formatted_results = []
		for result in results:
			if len(result) == 3:
				formatted_results.append({
					'image': os.path.basename(result[0]),
					'predicted_class': result[1],
					'confidence': float(result[2])
				})
			else:
				formatted_results.append({
					'image': os.path.basename(result[0]),
					'error': 'Failed to process image'
				})

		return jsonify({
			'message': 'Batch classification completed',
			'classes': class_list,
			'results': formatted_results
		}), 200

	except Exception as e:
		error_trace = traceback.format_exc()
		print(f"Error Traceback: {error_trace}")
		return jsonify({'error': f'Error processing image: {str(e)}', 'traceback': error_trace}), 500


@app.route('/', methods=['GET'])
async def index():
	return await render_template('index.html')

@app.route('/authform', methods=['GET', 'POST'])
async def authform():
	try:
		if request.method == 'POST':
			form_data = await request.form
			username = form_data.get('username')
			token = form_data.get('token')

			# Check if the user exists in the database
			user_data = await db_manager.get_user_by_email(username)

			if user_data:
				stored_token = user_data[2]  # Assuming token is in the third column

				if stored_token == token:
					return jsonify({
						'message': 'User authenticated',
						'username': username,
						'token': token
					}), 200
				else:
					return jsonify({'error': 'Invalid token'}), 401
			else:
				return jsonify({'error': 'User not found'}), 404
		else:
			return await render_template('index.html')

	except Exception as e:
		error_trace = traceback.format_exc()
		print(f"Error Traceback: {error_trace}")
		return jsonify({'error': str(e), 'traceback': error_trace}), 500

@app.route('/register', methods=['GET', 'POST'])
async def register():
	if request.method == 'GET':
		# Serve the Create New User page
		return await render_template('create_user.html')
	elif request.method == 'POST':
		# Handle form submission
		try:
			data = await request.json
			email = data.get('email')
			auth_token = data.get('auth_token')
			token_expiry = data.get('token_expiry')

			# Validate required fields
			if not email or not auth_token or not token_expiry:
				return jsonify({'error': 'All fields are required'}), 400

			# Create user in the database
			user_created = await db_manager.create_user(email, auth_token, token_expiry)

			if user_created:
				return jsonify({'message': 'User registered successfully'}), 201
			else:
				return jsonify({'error': 'Failed to register user'}), 500

		except Exception as e:
			error_trace = traceback.format_exc()
			print(f"Error Traceback: {error_trace}")
			return jsonify({'error': str(e), 'traceback': error_trace}), 500

import smtplib
from email.mime.text import MIMEText
from quart import request, jsonify

# Email configuration (update with your SMTP server details)
SMTP_SERVER = 'localhost'
SMTP_PORT = 1025
SMTP_USERNAME = ''
SMTP_PASSWORD = ''

FROM_EMAIL = 'tdrsvr@localhost.com'

@app.route('/send-token', methods=['POST'])
async def send_token():
    try:
        data = await request.json
        email = data.get('email')
        token = data.get('token')

        if not email or not token:
            return jsonify({'error': 'Email and token are required'}), 400

        # Create the email message
        subject = 'Your Verification Token'
        body = f'Your verification token is: {token}'
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = FROM_EMAIL
        msg['To'] = email

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            #server.starttls() - Renable in production !!***
            #server.login(SMTP_USERNAME, SMTP_PASSWORD) # Failed to send token: SMTP AUTH extension not supported by server. (local temp fix is comment)
            server.sendmail(FROM_EMAIL, [email], msg.as_string())

        return jsonify({'message': 'Token sent successfully'}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to send token: {str(e)}'}), 500

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5000)