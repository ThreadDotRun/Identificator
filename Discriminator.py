from quart import Quart, request, render_template_string
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
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            # For now, just return success message
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
            # TODO: Implement actual classification process
            # For now, return placeholder HTML report
            html_report = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Classification Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .report { max-width: 800px; margin: 0 auto; }
                    .result { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
                </style>
            </head>
            <body>
                <div class="report">
                    <h1>Classification Report</h1>
                    <div class="result">
                        <h2>Image: {{filename}}</h2>
                        <p>Classification results will appear here</p>
                    </div>
                </div>
            </body>
            </html>
            """
            return await render_template_string(html_report, filename=filename)
            
        except Exception as e:
            return {'error': f'Error processing image: {str(e)}'}, 500
        
    return {'error': 'Unknown error occurred'}, 500

@app.route('/', methods=['GET'])
async def index():
    """
    Simple index page with upload forms
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Model Interface</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-form { margin: 20px 0; padding: 20px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ML Model Interface</h1>
            
            <div class="upload-form">
                <h2>Fine-tune Model</h2>
                <form action="/finetune" method="post" enctype="multipart/form-data">
                    <p>Upload ZIP file containing image folders:</p>
                    <input type="file" name="file" accept=".zip">
                    <input type="submit" value="Upload and Fine-tune">
                </form>
            </div>
            
            <div class="upload-form">
                <h2>Classify Image</h2>
                <form action="/classify" method="post" enctype="multipart/form-data">
                    <p>Upload image for classification:</p>
                    <input type="file" name="file" accept="image/*">
                    <input type="submit" value="Upload and Classify">
                </form>
            </div>
        </div>
    </body>
    </html>
    """
    return await render_template_string(html)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)