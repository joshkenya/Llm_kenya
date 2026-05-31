"""
Web UI for LLM Model
Flask-based interface for interactive chat with the model
"""

from flask import Flask, render_template, request, jsonify
import os
from pathlib import Path
from chat import ChatBot
from model import Config

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize chatbot
MODEL_PATH = os.environ.get('MODEL_PATH', './checkpoints/final_model')
chatbot = ChatBot(model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None)

# Store conversation history in session-like manner
conversations = {}
current_session_id = None


@app.route('/')
def index():
    """Serve the main chat page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Generate response
        response = chatbot.chat(user_message)
        
        return jsonify({
            'user_message': user_message,
            'bot_response': response,
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = chatbot.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        return jsonify({
            'history': chatbot.conversation_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        chatbot.conversation_history = []
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-checkpoint', methods=['POST'])
def save_checkpoint():
    """Save model checkpoint"""
    try:
        data = request.json
        checkpoint_name = data.get('name', 'checkpoint')
        checkpoint_path = f'./checkpoints/{checkpoint_name}'
        
        chatbot.save_checkpoint(checkpoint_path)
        
        return jsonify({
            'success': True,
            'message': f'Checkpoint saved to {checkpoint_path}',
            'path': checkpoint_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load-checkpoint', methods=['POST'])
def load_checkpoint():
    """Load model checkpoint"""
    try:
        data = request.json
        checkpoint_path = data.get('path')
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return jsonify({'error': 'Invalid checkpoint path'}), 400
        
        chatbot.load_checkpoint(checkpoint_path)
        
        return jsonify({
            'success': True,
            'message': f'Checkpoint loaded from {checkpoint_path}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/available-checkpoints', methods=['GET'])
def available_checkpoints():
    """List available checkpoints"""
    try:
        checkpoint_dir = Path('./checkpoints')
        
        if not checkpoint_dir.exists():
            return jsonify({'checkpoints': []})
        
        checkpoints = [
            item.name for item in checkpoint_dir.iterdir()
            if item.is_dir()
        ]
        
        return jsonify({'checkpoints': checkpoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create necessary directories
    Path('./checkpoints').mkdir(exist_ok=True)
    Path('./templates').mkdir(exist_ok=True)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
