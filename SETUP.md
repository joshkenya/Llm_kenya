# LLM Kenya - Configuration & Setup Guide

## Quick Start

### Option 1: Using the Shell Script (Linux/Mac)

```bash
# Make script executable
chmod +x run.sh

# Run web UI
./run.sh web

# Run interactive chat
./run.sh chat

# View model information
./run.sh model-info
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run web UI
python3 app.py

# Or run chat
python3 chat.py
```

### Option 3: Google Colab Training

1. Go to https://colab.research.google.com
2. Click "File" → "Open Notebook"
3. Go to "GitHub" tab
4. Enter: `joshkenya/Llm_kenya`
5. Select `train_colab.ipynb`
6. Run all cells

## File Structure

```
Llm_kenya/
├── model.py                    # Core transformer implementation
│   ├── Config                 # Model configuration
│   ├── LayerNorm              # Layer normalization
│   ├── Embedding              # Token & position embeddings
│   ├── MultiHeadAttention     # Self-attention mechanism
│   ├── FeedForward            # FFN layers
│   ├── TransformerLayer       # Single transformer block
│   ├── TransformerModel       # Complete 60M model
│   ├── Tokenizer              # Simple BPE tokenizer
│   └── Trainer                # Training utility
│
├── chat.py                     # CLI chat interface
│   ├── ChatBot                # Chat interface class
│   └── main()                 # Entry point
│
├── app.py                      # Flask web application
│   ├── /api/chat              # Chat endpoint
│   ├── /api/model-info        # Model info endpoint
│   ├── /api/history           # Conversation history
│   ├── /api/save-checkpoint   # Save model
│   └── /api/load-checkpoint   # Load model
│
├── train_colab.ipynb          # Google Colab training notebook
│   ├── Setup & Installation
│   ├── Repository cloning
│   ├── Model initialization
│   ├── Dataset loading
│   ├── Training loop
│   ├── Checkpoint saving
│   └── Generation testing
│
├── templates/
│   └── index.html             # Web UI frontend
│       ├── Chat interface
│       ├── Model info display
│       ├── Message history
│       └── Responsive design
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── run.sh                     # Startup script
└── README.md                 # Full documentation
```

## Model Specifications

| Aspect | Value |
|--------|-------|
| **Total Parameters** | ~60M |
| **Vocabulary Size** | 50,257 |
| **Hidden Dimension** | 768 |
| **Number of Layers** | 12 |
| **Attention Heads** | 12 |
| **Feed-Forward Dimension** | 3,072 |
| **Max Sequence Length** | 2,048 |
| **Activation Function** | GELU |

## Training Configuration

Default training parameters (adjustable in notebook):

```python
num_epochs = 1
batch_size = 2
learning_rate = 1e-4
checkpoint_interval = 100
max_sequence_length = 256
```

### Adjusting Parameters

In `train_colab.ipynb`, modify these cells:

```python
# For longer training
num_epochs = 3

# For larger batches (if GPU memory allows)
batch_size = 8

# For faster learning
learning_rate = 5e-4

# For more frequent checkpoints
checkpoint_interval = 50
```

## Checkpoint System

### Checkpoint Structure

```
checkpoints/
└── my_model/
    ├── config.json           # Model configuration
    └── model_params.npz      # All parameters (compressed)
```

### Saving Checkpoints

```python
from chat import ChatBot

chatbot = ChatBot()
chatbot.save_checkpoint('./checkpoints/my_model')
```

### Loading Checkpoints

```python
chatbot = ChatBot(model_path='./checkpoints/my_model')
```

## Deployment

### Local Deployment

```bash
# Run web UI
python3 app.py

# Accessible at http://localhost:5000
```

### Cloud Deployment (Heroku)

```bash
# Add Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
git push heroku main
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "app.py"]
```

## API Documentation

### Chat Endpoint

```
POST /api/chat
Content-Type: application/json

Request:
{
  "message": "Your message here"
}

Response:
{
  "user_message": "Your message here",
  "bot_response": "Generated response",
  "success": true
}
```

### Model Info Endpoint

```
GET /api/model-info

Response:
{
  "total_parameters": 60000000,
  "parameters_millions": 60.0,
  "hidden_size": 768,
  "num_layers": 12,
  "vocab_size": 50257,
  "max_sequence_length": 2048
}
```

### History Endpoint

```
GET /api/history

Response:
{
  "history": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi there!"
    }
  ]
}
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Out of memory" during training

**Solution:** Reduce batch size or sequence length
```python
batch_size = 1  # Reduce from 2
max_sequence_length = 128  # Reduce from 256
```

### Issue: Model generation is slow

**Solution:** 
- Use GPU acceleration
- Reduce sequence length
- Use smaller batch sizes for generation

### Issue: Web UI not loading

**Solution:** Check Flask is running
```bash
# Should see: Running on http://0.0.0.0:5000/
python3 app.py
```

## Performance Tips

1. **Use GPU**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   python3 train_colab.ipynb  # In Colab
   ```

2. **Optimize Batch Size**
   - Larger batches = faster training
   - Constrained by GPU memory

3. **Learning Rate Schedule**
   - Start with 1e-4
   - Reduce to 1e-5 after early epochs

4. **Gradient Accumulation**
   - Simulate larger batches with smaller GPU memory
   - Accumulate gradients over multiple steps

## System Requirements

### Minimum
- Python 3.7+
- 4GB RAM
- 2GB disk space

### Recommended
- Python 3.8+
- 8GB RAM
- 10GB disk space
- GPU with 4GB+ VRAM

### For Google Colab
- Free account
- GPU acceleration (enabled by default)

## Environment Variables

```bash
# Flask configuration
FLASK_ENV=development  # or production
FLASK_DEBUG=1

# Model configuration
MODEL_PATH=./checkpoints/final_model

# Port configuration
PORT=5000
```

## Maintenance

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Clean Up Old Checkpoints

```bash
rm -rf checkpoints/checkpoint_step_*
```

### Monitor Training

```bash
# Watch training logs
tail -f training.log
```

## Support & Community

- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Read README.md for details

## Next Steps

1. ✅ Clone the repository
2. ✅ Install dependencies
3. ✅ Run the model (chat or web UI)
4. ✅ Train in Google Colab
5. ✅ Deploy to production

## Resources

- [Transformer Architecture Paper](https://arxiv.org/abs/1706.03762)
- [NumPy Documentation](https://numpy.org/doc/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [HuggingFace Datasets](https://huggingface.co/datasets)

---

**Ready to get started? Run `./run.sh web` or `python3 app.py`! 🚀**
