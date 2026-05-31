"""
Chat interface for the LLM model
Provides interactive conversation capabilities
"""

import numpy as np
from model import TransformerModel, Config, Tokenizer, Trainer
from pathlib import Path
import os


class ChatBot:
    """Interactive chatbot interface"""
    def __init__(self, model_path: str = None, config: Config = None):
        self.config = config or self._get_default_config()
        self.model = TransformerModel(self.config)
        self.tokenizer = Tokenizer(self.config.vocab_size)

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model.load_checkpoint(model_path)
            print("Model loaded successfully!")
        else:
            print("Using randomly initialized model. Train with train_colab.ipynb for better results.")

        self.conversation_history = []

    @staticmethod
    def _get_default_config() -> Config:
        """Get default model configuration"""
        return Config(
            vocab_size=50257,
            max_position_embeddings=2048,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )

    def _generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate response from prompt"""
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)

        # Limit to max position length
        tokens = tokens[-self.config.max_position_embeddings + 1:]
        current_length = len(tokens)

        # Generate tokens
        for _ in range(max_length):
            # Pad to ensure proper batch dimension
            if current_length < self.config.max_position_embeddings:
                padded_tokens = tokens + [self.tokenizer.pad_token_id] * (
                    self.config.max_position_embeddings - current_length
                )
            else:
                padded_tokens = tokens[-self.config.max_position_embeddings:]

            # Forward pass
            input_ids = np.array([padded_tokens])
            logits = self.model.forward(input_ids)

            # Get last token logits
            last_logits = logits[0, current_length - 1, :]

            # Apply temperature
            last_logits = last_logits / temperature

            # Softmax
            exp_logits = np.exp(last_logits - np.max(last_logits))
            probs = exp_logits / np.sum(exp_logits)

            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)

            # Check for EOS token
            if next_token == self.tokenizer.eos_token_id:
                break

            tokens.append(next_token)
            current_length += 1

            # Update sequence length tracking
            if current_length > self.config.max_position_embeddings:
                current_length = self.config.max_position_embeddings

        # Decode response
        response = self.tokenizer.decode(tokens)
        return response

    def chat(self, user_input: str) -> str:
        """Process user input and generate response"""
        # Add to history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Generate response
        response = self._generate_response(user_input, max_length=100, temperature=0.7)

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def interactive_chat(self):
        """Run interactive chat session"""
        print("\n" + "=" * 50)
        print("Welcome to LLM Chat!")
        print("Type 'quit' to exit")
        print("Type 'history' to see conversation history")
        print("Type 'clear' to clear history")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "history":
                    self._print_history()
                    continue

                if user_input.lower() == "clear":
                    self.conversation_history = []
                    print("History cleared.")
                    continue

                response = self.chat(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _print_history(self):
        """Print conversation history"""
        if not self.conversation_history:
            print("No conversation history.")
            return

        print("\n" + "=" * 50)
        print("Conversation History:")
        print("=" * 50)
        for msg in self.conversation_history:
            role = msg["role"].upper()
            content = msg["content"]
            print(f"{role}: {content}\n")
        print("=" * 50 + "\n")

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "total_parameters": self.model.total_params,
            "parameters_millions": round(self.model.total_params / 1e6, 2),
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_hidden_layers,
            "vocab_size": self.config.vocab_size,
            "max_sequence_length": self.config.max_position_embeddings,
        }

    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint"""
        self.model.save_checkpoint(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            self.model.load_checkpoint(checkpoint_path)
            print(f"Model loaded from {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")


def main():
    """Main entry point"""
    # Initialize chatbot
    chatbot = ChatBot()

    # Print model info
    print("\n" + "=" * 50)
    print("Model Information:")
    info = chatbot.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

    # Start interactive chat
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()
