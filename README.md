# LLM Kenya - 60M Parameter Transformer Model

A production-ready Large Language Model (LLM) built from scratch using only **Python and NumPy**. This is a 60M parameter transformer model trained on free datasets from HuggingFace.

## 🚀 Features

- **60M Parameter Transformer Model** - Built from scratch with pure Python and NumPy
- **Production Ready** - Checkpoint saving/loading, proper error handling
- **No Heavy Dependencies** - Only NumPy for computation (+ Flask for web UI)
- **Google Colab Compatible** - Includes notebook for training on free GPU
- **Multiple Interfaces**:
  - 💬 Interactive CLI chat (`chat.py`)
  - 🌐 Web UI with Flask (`app.py`)
  - 📓 Jupyter Notebook for training (`train_colab.ipynb`)
- **Checkpoint System** - Save and load trained model weights
- **HuggingFace Datasets** - Uses free WikiText-2 dataset for training

## 📊 Model Architecture

```
Configuration:
- Vocabulary Size: 50,257
- Hidden Size: 768
- Number of Layers: 12
- Attention Heads: 12
- Intermediate Size (FFN): 3,072
- Max Sequence Length: 2,048
- Total Parameters: ~60M
```

### Architecture Components

1. **Token & Position Embeddings**
   - Token embedding layer for vocabulary
   - Positional encoding for sequence position awareness

2. **Multi-Head Self Attention**
   - 12 attention heads
   - Scaled dot-product attention
   - Causal masking for autoregressive generation

3. **Feed-Forward Networks**
   - Two dense layers with GELU activation
   - Separate FFN for each transformer layer

4. **Layer Normalization & Residual Connections**
   - Layer normalization after attention and FFN
   - Residual connections for improved gradient flow

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- `numpy>=1.21.0` - Core numerical computations
- `Flask>=2.0.0` - Web UI framework
- `datasets>=2.0.0` - HuggingFace datasets
- `transformers>=4.0.0` - Tokenization utilities
- `torch>=1.9.0` - GPU acceleration (optional)

## 🎯 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/joshkenya/Llm_kenya.git
cd Llm_kenya
pip install -r requirements.txt
```

### 2. Run Interactive Chat

```bash
python chat.py
```

Example:
```
You: Hello, how are you?
Assistant: [generates response based on training]

You: Tell me about machine learning
Assistant: [generates response]

Type 'quit' to exit, 'history' to see conversation, 'clear' to clear history
```

### 3. Run Web UI

```bash
python app.py
```

Then open browser and navigate to `http://localhost:5000`

### 4. Train Model in Google Colab

1. Open `train_colab.ipynb` in Google Colab
2. Run all cells in order
3. The notebook will:
   - Clone this repository
   - Download WikiText-2 dataset
   - Train the model
   - Save checkpoints
   - Allow you to test generation

## 🏋️ Training

### Local Training

```python
from model import TransformerModel, Config, Trainer, Tokenizer
from datasets import load_dataset
import numpy as np

# Initialize model
config = Config()
model = TransformerModel(config)
trainer = Trainer(model, learning_rate=1e-4)

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-2')

# Training loop
for epoch in range(num_epochs):
    for batch_input_ids, batch_target_ids in data_loader:
        loss = trainer.train_step(batch_input_ids, batch_target_ids)
        
    # Save checkpoint
    model.save_checkpoint(f'./checkpoints/epoch_{epoch}')
```

### In Google Colab

1. Upload this repo to Colab
2. Run the provided `train_colab.ipynb` notebook
3. Customize training parameters as needed
4. Download trained checkpoints

## 💾 Checkpoint Management

### Save Checkpoint

```python
from chat import ChatBot

chatbot = ChatBot()
chatbot.save_checkpoint('./checkpoints/my_model')
```

### Load Checkpoint

```python
chatbot = ChatBot(model_path='./checkpoints/my_model')
```

The checkpoint system saves:
- Model configuration (JSON)
- All trainable parameters (NumPy compressed format)
- Layer weights and biases

## 🌐 Web UI API

The Flask application provides REST endpoints:

### Chat Endpoint
```bash
POST /api/chat
Content-Type: application/json

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

### Model Info
```bash
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

### Other Endpoints
- `GET /api/history` - Get conversation history
- `POST /api/clear-history` - Clear conversation
- `POST /api/save-checkpoint` - Save model
- `POST /api/load-checkpoint` - Load model
- `GET /api/available-checkpoints` - List checkpoints

## 📁 Project Structure

```
Llm_kenya/
├── model.py                    # Core transformer model
├── chat.py                     # CLI chat interface
├── app.py                      # Flask web application
├── train_colab.ipynb          # Google Colab training notebook
├── templates/
│   └── index.html             # Web UI frontend
├── requirements.txt            # Python dependencies
├── checkpoints/               # Model checkpoints directory
│   └── final_model/
│       ├── config.json
│       └── model_params.npz
└── README.md                  # This file
```

## 🔧 Configuration

Create a custom model by modifying the Config class:

```python
from model import Config, TransformerModel

config = Config(
    vocab_size=50257,
    max_position_embeddings=2048,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

model = TransformerModel(config)
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 50257 | Vocabulary size |
| `max_position_embeddings` | 2048 | Maximum sequence length |
| `hidden_size` | 768 | Hidden layer dimension |
| `num_hidden_layers` | 12 | Number of transformer layers |
| `num_attention_heads` | 12 | Number of attention heads |
| `intermediate_size` | 3072 | Feed-forward layer size |
| `hidden_dropout_prob` | 0.1 | Dropout probability |
| `attention_probs_dropout_prob` | 0.1 | Attention dropout probability |

## 🎓 Learning Resources

### Understanding the Implementation

1. **Embeddings** - Token and positional embeddings
2. **Multi-Head Attention** - Self-attention mechanism
3. **Feed-Forward Networks** - Two-layer dense networks
4. **Transformer Layers** - Stacking attention + FFN
5. **Training Loop** - Forward pass and loss computation

### Key Concepts

- **Self-Attention**: Each token attends to all other tokens
- **Multi-Head**: Multiple attention heads capture different patterns
- **Residual Connections**: Help gradients flow through deep networks
- **Layer Normalization**: Stabilizes training
- **Causal Masking**: Prevents attending to future tokens

## 🚀 Performance Tips

1. **Use GPU**: Set `CUDA_VISIBLE_DEVICES` for GPU acceleration
2. **Batch Size**: Adjust based on available memory
3. **Learning Rate**: Start with 1e-4 and adjust as needed
4. **Checkpointing**: Save regularly during training

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing

### Slow Training
- Enable GPU acceleration
- Use larger batch sizes (if memory allows)
- Reduce number of training steps

### Model Not Responding
- Check if model is loaded: `chatbot.model`
- Verify checkpoint files exist
- Check memory and disk space

## 📝 Example Usage

### Interactive Chat Session

```bash
$ python chat.py

==================================================
Welcome to LLM Chat!
Type 'quit' to exit
Type 'history' to see conversation history
Type 'clear' to clear history
==================================================

You: What is artificial intelligence?
Assistant: [Model generates response...]

You: Tell me more about neural networks
Assistant: [Model generates response...]

You: quit
Goodbye!
```

### Python Script Usage

```python
from chat import ChatBot

# Initialize chatbot
bot = ChatBot(model_path='./checkpoints/final_model')

# Print model info
print(bot.get_model_info())

# Generate responses
response = bot.chat("Hello, what is machine learning?")
print(response)

# Access conversation history
print(bot.conversation_history)

# Save model
bot.save_checkpoint('./checkpoints/v2')
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Submit bug reports
- Suggest improvements
- Contribute code
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Python and NumPy
- Training data from HuggingFace Datasets (WikiText-2)
- Inspired by modern transformer architectures

## 📞 Support

For issues and questions:
1. Check troubleshooting section
2. Review code comments
3. Check existing GitHub issues
4. Create a new issue if needed

## 📚 References

- Attention is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- HuggingFace Documentation
- NumPy Documentation

---

**Happy Training! 🚀**

Built with ❤️ for the LLM Kenya project
