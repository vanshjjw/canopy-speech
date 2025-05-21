import torch
import torch.nn as nn
from transformers import WhisperModel, LlamaForCausalLM, LlamaTokenizer
from typing import Optional, Tuple

class Adaptor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list):
        super().__init__()
        
        layers = []
        current_dimension = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dimension, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_dimension = hidden_dim
        self.mlp_output_dimension = current_dimension
            
        self.mlp = nn.Sequential(*layers)
        self.projection = nn.Linear(self.mlp_output_dimension, output_dim)
        
        self.pre_norm = nn.LayerNorm(input_dim)
        self.post_norm = nn.LayerNorm(output_dim)
        self.skip_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.pre_norm(x) # Shape: [B, S, input_dim]
        
        mlp_features = self.mlp(x_norm) 
        projected_features = self.projection(mlp_features) 
        
        skip = self.skip_connection(x) 
        outputs = self.post_norm(projected_features + skip)     
        
        return outputs





class SpeechToTextModel(nn.Module):
    def __init__(
        self,
        whisper_model_name: str,
        llama_model_name: str,
        hidden_dims: list = None,
        train_whisper: bool = True,
    ):
        super().__init__()

        # Initialize Whisper encoder
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        if not train_whisper:
            for param in self.whisper.parameters():
                param.requires_grad = False
        
        # Initialize Llama model
        self.llama = LlamaForCausalLM.from_pretrained(llama_model_name)
        # self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)

        if hidden_dims is None:
            hidden_dims = [2048, 1024, 2048]

        self.projection_dim = self.llama.config.hidden_size

        self.adaptor = Adaptor(
            input_dim=self.whisper.config.hidden_size,
            output_dim=self.projection_dim,
            hidden_dims=hidden_dims
        )
        
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        whisper_outputs = self.whisper.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        projected_features = self.adaptor(whisper_outputs.last_hidden_state)
        
        llama_outputs = self.llama(
            inputs_embeds=projected_features,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return llama_outputs.logits, llama_outputs.loss


    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        **kwargs
    ) -> torch.Tensor:

        whisper_outputs = self.whisper.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        projected_features = self.adaptor(whisper_outputs.last_hidden_state)
        
        generated_ids = self.llama.generate(
            inputs_embeds=projected_features,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs
        )
        
        return generated_ids 