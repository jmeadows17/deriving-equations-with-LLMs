import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers import EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=1)

# Custom configuration class (optional but useful)
class LlamaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=None,      # Adjust as needed
        max_position_embeddings=1024,
        embedding_dim=4096,
        num_heads=32,            # 32 for llama-8b
        num_layers=32,           # Number of transformer blocks
        hidden_dim=None,       # Typically 4 * embedding_dim
        num_groups=8,            # For Grouped Query Attention
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or embedding_dim * 4
        self.num_groups = num_groups

# RMSNorm implementation
def rms_norm(x, weight=None, eps=1e-8):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    normed_x = x / rms
    if weight is not None:
        normed_x = normed_x * weight
    return normed_x

# SwiGLU activation
def swiGLU(x):
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * torch.nn.functional.silu(x2)

# Function to generate RoPE frequencies
def get_rope_frequencies(head_dim, base=10000, device='cuda'):
    dim = torch.arange(0, head_dim, 2, device=device)
    freqs = 1.0 / (base ** (dim.float() / head_dim))
    return freqs  # Shape: [head_dim // 2]

def apply_rope(q_or_k, freqs):
    batch_size, seq_len, num_heads, head_dim = q_or_k.shape
    half_dim = head_dim // 2
    q_or_k = q_or_k.contiguous().view(batch_size, seq_len, num_heads, half_dim, 2)
    positions = torch.arange(seq_len, device=q_or_k.device, dtype=freqs.dtype)
    angles = positions[:, None] * freqs[None, :]
    angles = angles[None, :, None, :, None]
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    q_or_k_rotated = torch.empty_like(q_or_k)
    q_or_k_rotated[..., 0] = q_or_k[..., 0] * cos_angles[..., 0] - q_or_k[..., 1] * sin_angles[..., 0]
    q_or_k_rotated[..., 1] = q_or_k[..., 0] * sin_angles[..., 0] + q_or_k[..., 1] * cos_angles[..., 0]
    q_or_k_rotated = q_or_k_rotated.view(batch_size, seq_len, num_heads, head_dim)
    return q_or_k_rotated

# Define a function to apply templates to conversations
def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

# Custom model class
class LlamaModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize weights as per Llama-2 guidelines
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Embedding
        M = self.embedding_layer(input_ids)

        # Transformer blocks
        for layer in self.layers:
            M = layer(M, attention_mask)

        # Final layer norm
        M = rms_norm(M)

        # Language modeling head
        logits = self.lm_head(M)

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits}



class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        embedding_dim = config.embedding_dim
        num_heads = config.num_heads
        head_dim = embedding_dim // num_heads
        num_groups = config.num_groups
        hidden_dim = config.hidden_dim

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_groups = num_groups

        # Build the head_to_group mapping
        base_heads_per_group = self.num_heads // self.num_groups
        remainder_heads = self.num_heads % self.num_groups

        head_to_group = []
        for group_idx in range(num_groups):
            group_size = base_heads_per_group + (1 if group_idx < remainder_heads else 0)
            head_to_group.extend([group_idx] * group_size)
        self.register_buffer('head_to_group', torch.tensor(head_to_group, dtype=torch.long))

        # RoPE frequencies
        self.register_buffer('freqs', get_rope_frequencies(head_dim, device='cuda'))

        # Grouped Query Attention parameters
        self.W_Q = nn.Linear(embedding_dim, num_heads * head_dim, bias=False)
        self.W_K_grouped = nn.Linear(embedding_dim, num_groups * head_dim, bias=False)
        self.W_V_grouped = nn.Linear(embedding_dim, num_groups * head_dim, bias=False)
        self.W_O = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Feed-forward network parameters
        self.W_1 = nn.Linear(embedding_dim, hidden_dim * 2)
        self.W_2 = nn.Linear(hidden_dim, embedding_dim)

    def self_attention(self, M, attention_mask):
        batch_size, sequence_length, embedding_dim = M.size()

        # Ensure freqs is on correct device
        freqs = self.freqs.to(M.device)
        head_to_group = self.head_to_group.to(M.device)

        # Compute Q, K, V projections
        Q = self.W_Q(M).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        K = self.W_K_grouped(M).view(batch_size, sequence_length, self.num_groups, self.head_dim)
        V = self.W_V_grouped(M).view(batch_size, sequence_length, self.num_groups, self.head_dim)

        # Map K and V to per-head versions using head_to_group
        K_per_head = K[:, :, head_to_group, :]
        V_per_head = V[:, :, head_to_group, :]

        # Apply RoPE embeddings
        Q = apply_rope(Q, freqs)
        K_per_head = apply_rope(K_per_head, freqs)

        # Transpose for batched matrix multiplication
        Q = Q.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        K_per_head = K_per_head.permute(0, 2, 3, 1)  # [batch_size, num_heads, head_dim, seq_len]
        V_per_head = V_per_head.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K_per_head) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            # Convert attention_mask to [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :].to(attn_scores.dtype)
            # Use a large negative value for masking (not necessarily -inf for numerical stability)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-1e9'))

        # Apply causal mask
        causal_mask = torch.triu(torch.ones((sequence_length, sequence_length), device=attn_scores.device), diagonal=1)
        causal_mask = causal_mask.bool()
        attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], float('-1e9'))

        # Apply softmax to get attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, V_per_head)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, -1)
        
        return attn_output


    def feed_forward(self, M):
        ffn_input = self.W_1(M)
        F_1 = swiGLU(ffn_input)
        F_2 = self.W_2(F_1)
        return F_2

    def forward(self, M, attention_mask=None):
        
        # Pre-normalisation before attention
        M_norm = rms_norm(M)
        O = self.self_attention(M_norm, attention_mask)
        M = M + O

        # Pre-normalisation before feed-forward
        M_norm = rms_norm(M)
        F = self.feed_forward(M_norm)
        M = M + F
        return M
    
# Prepare the tokenizer
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
max_seq_length = 1024
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
tokenizer.pad_token = tokenizer.eos_token  # Ensure the pad token is set

# Configure the tokenizer with chat templates
tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

# Load the dataset and split into train and validation
dataset = load_dataset('arrow', data_files='training_data.arrow', split="train")
dataset = dataset.map(apply_template, batched=True)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print(train_test_split)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "conversations"],
)

tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "conversations"],
)

# Prepare labels
def prepare_labels(examples):
    labels = []
    for input_ids in examples["input_ids"]:
        # Shift labels one token to the left
        shifted_labels = input_ids[1:] + [tokenizer.pad_token_id]
        labels.append([id if id != tokenizer.pad_token_id else -100 for id in shifted_labels])
    examples["labels"] = labels
    return examples


tokenized_train_dataset = tokenized_train_dataset.map(prepare_labels, batched=True)
tokenized_eval_dataset = tokenized_eval_dataset.map(prepare_labels, batched=True)

# Initialize the model
config = LlamaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=max_seq_length,
    embedding_dim=1024,
    num_heads=16,
    num_layers=16,
    num_groups=8,
)
model = LlamaModel(config)

# parameters
batch_size = 1
learning_rate = 5e-5
epochs = 25

# Update training arguments
training_args = TrainingArguments(
    learning_rate=learning_rate,
    lr_scheduler_type="linear",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=16,
    num_train_epochs=epochs,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    optim="adamw_8bit",
    weight_decay=0.01,
    output_dir="output",
    seed=42,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    callbacks=[early_stopping_callback]
)

# Start training
trainer.train()
trainer.save_model('best_model')
tokenizer.save_pretrained('best_model')
