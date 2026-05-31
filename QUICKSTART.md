# LLM Kenya - Quick Start Guide

## 🎯 Get Started in 2 Minutes

### Step 1: Clone Repository
```bash
git clone https://github.com/joshkenya/Llm_kenya.git
cd Llm_kenya
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Choose Your Interface

#### Option A: Web UI (Recommended)
```bash
python3 app.py
# Open: http://localhost:5000
```

#### Option B: Interactive Chat
```bash
python3 chat.py
```

#### Option C: Use Shell Script
```bash
chmod +x run.sh
./run.sh web      # Web UI
./run.sh chat     # Chat
./run.sh model-info  # Model info
```

## 📚 Training in Google Colab

1. Open https://colab.research.google.com
2. Click **File** → **Open Notebook**
3. Select **GitHub** tab
4. Enter: `joshkenya/Llm_kenya`
5. Open **train_colab.ipynb**
6. Run all cells (GPU enabled by default)

## 🗂️ Key Files

| File | Purpose |
|------|---------|
| `model.py` | 60M parameter transformer model |
| `chat.py` | Interactive CLI chat interface |
| `app.py` | Flask web application |
| `train_colab.ipynb` | Google Colab training notebook |
| `templates/index.html` | Web UI frontend |

## 💾 Model Checkpoints

### Save Model
```python
from chat import ChatBot
bot = ChatBot()
bot.save_checkpoint('./checkpoints/my_model')
```

### Load Model
```python
bot = ChatBot(model_path='./checkpoints/my_model')
```

## 🌐 Web UI Features

- ✅ Real-time chat interface
- ✅ Message history
- ✅ Model information display
- ✅ Responsive design
- ✅ Easy to use

## 🏋️ Training Parameters

In `train_colab.ipynb`, adjust:

```python
num_epochs = 1              # Number of training epochs
batch_size = 2              # Batch size (1-8 recommended)
learning_rate = 1e-4        # Learning rate
checkpoint_interval = 100   # Save every N steps
```

## 📊 Model Stats

- **Parameters**: 60 Million
- **Hidden Size**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Vocab Size**: 50,257
- **Max Sequence**: 2,048 tokens

## 🔗 API Endpoints

```bash
# Chat
POST /api/chat
{ "message": "Your message" }

# Model Info
GET /api/model-info

# Conversation History
GET /api/history

# Clear History
POST /api/clear-history
```

## 🐛 Common Issues

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: Port already in use
```bash
python3 app.py --port 5001
```

### Issue: Out of memory during training
Reduce batch size in notebook:
```python
batch_size = 1
```

## 📖 Full Documentation

- See `README.md` for complete documentation
- See `SETUP.md` for detailed setup instructions
- See code comments in `model.py` for implementation details

## 🚀 Next Steps

1. ✅ Run the model locally
2. ✅ Test the web UI
3. ✅ Train in Google Colab
4. ✅ Deploy to production (optional)

## 💡 Tips

- **First time?** Start with `./run.sh web`
- **Training?** Use Google Colab for free GPU
- **Production?** Save checkpoints frequently
- **Help?** Check README.md or SETUP.md

## 📞 Support

- 📖 Documentation: README.md
- ⚙️ Setup Help: SETUP.md
- 🐛 Issues: GitHub Issues
- 💬 Questions: GitHub Discussions

---

**Ready to go? Run `python3 app.py` now! 🚀**

For detailed information, see [README.md](README.md) and [SETUP.md](SETUP.md)
