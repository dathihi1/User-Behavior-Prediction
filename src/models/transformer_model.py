"""Transformer-based multi-output classifier for sequence data."""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerNetwork(nn.Module):
    """Transformer network for multi-output classification."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: List[int],
        dropout: float = 0.1,
        max_len: int = 512,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Output heads for each target
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, nc),
            )
            for nc in num_classes
        ])

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        batch_size = x.size(0)

        # Create padding mask
        padding_mask = (x == self.padding_idx)

        # Embedding
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat([cls_tokens, embedded], dim=1)

        # Extend padding mask for CLS token
        cls_mask = torch.zeros(batch_size, 1, device=x.device, dtype=torch.bool)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Transformer
        output = self.transformer(embedded, src_key_padding_mask=padding_mask)

        # Use CLS token output
        cls_output = output[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Multi-output heads
        outputs = [head(cls_output) for head in self.output_heads]

        return outputs


class TransformerMultiOutput(BaseModel):
    """Transformer-based multi-output classifier."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        batch_size: int = 256,
        epochs: int = 100,
        learning_rate: float = 0.0003,
        patience: int = 20,
        warmup_epochs: int = 3,
        device: Optional[str] = None,
        random_state: int = 42,
        name: str = "transformer_multi",
    ):
        super().__init__(name=name)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.num_classes = None
        self.history = {"train_loss": [], "val_loss": []}

    def _build_model(self, num_classes: List[int]) -> TransformerNetwork:
        """Build Transformer network."""
        model = TransformerNetwork(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            num_classes=num_classes,
            dropout=self.dropout,
            max_len=self.max_len,
        ).to(self.device)

        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_classes: Optional[List[int]] = None,
    ) -> "TransformerMultiOutput":
        """Train Transformer model.

        Args:
            num_classes: Number of classes per target column. Pass
                ``list(target_encoder.num_classes.values())`` to guarantee
                all known classes are covered even if some are absent from
                y_train (e.g. rare classes in small splits).
        """
        set_seed(self.random_state)
        logger.info(f"Training {self.name} on {X_train.shape[0]} samples...")
        logger.info(f"Using device: {self.device}")

        # Determine number of classes for each target.
        # Prefer the caller-supplied list (from TargetEncoder) so that rare
        # classes absent from this split are not silently dropped.
        if num_classes is not None:
            self.num_classes = list(num_classes)
        else:
            self.num_classes = [
                int(y_train[:, i].max()) + 1 for i in range(y_train.shape[1])
            ]
        logger.info(f"Number of classes per target: {self.num_classes}")

        # Build model
        self.model = self._build_model(self.num_classes)

        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        # LambdaLR: linear warmup then cosine decay — robust against ZeroDivisionError
        _total_steps = self.epochs * len(train_loader)
        _warmup_steps = min(self.warmup_epochs * len(train_loader), _total_steps)

        def _lr_lambda(step: int) -> float:
            if step < _warmup_steps:
                return step / max(1, _warmup_steps)
            progress = (step - _warmup_steps) / max(1, _total_steps - _warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Mixed precision training (AMP) for GPU speedup
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training (scheduler steps every batch inside _train_epoch)
            train_loss = self._train_epoch(train_loader, optimizer, criterion, scaler, use_amp, scheduler)
            self.history["train_loss"].append(train_loss)

            # Validation
            if val_loader:
                val_loss = self._validate_epoch(val_loader, criterion, use_amp)
                self.history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}")

        # Load best model state
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)

        self._fitted = True
        logger.info(f"Training completed for {self.name}")
        return self

    def _create_dataloader(
        self, X: np.ndarray, y: Optional[np.ndarray], shuffle: bool
    ) -> DataLoader:
        """Create PyTorch DataLoader with performance optimizations."""
        X_tensor = torch.LongTensor(X)
        if y is not None:
            y_tensor = torch.LongTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        # Performance optimizations
        use_cuda = self.device.type == "cuda"
        # Windows: num_workers > 0 can cause issues, use 0 for safety
        # Linux/Mac: use 4 workers for parallel data loading
        import platform
        if platform.system() == "Windows":
            num_workers = 0
        else:
            num_workers = 4 if not use_cuda else 2

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )

    def _train_epoch(
        self, loader: DataLoader, optimizer, criterion, scaler, use_amp: bool,
        scheduler=None,
    ) -> float:
        """Train for one epoch with optional mixed precision."""
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            X_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = self.model(X_batch)
                loss = sum(
                    criterion(outputs[i], y_batch[:, i])
                    for i in range(len(outputs))
                )

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # OneCycleLR steps every batch
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader, criterion, use_amp: bool = False) -> float:
        """Validate for one epoch with optional mixed precision."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = self.model(X_batch)
                    loss = sum(
                        criterion(outputs[i], y_batch[:, i])
                        for i in range(len(outputs))
                    )
                total_loss += loss.item()

        return total_loss / len(loader)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all 6 targets."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self.model.eval()
        loader = self._create_dataloader(X, None, shuffle=False)

        all_predictions = [[] for _ in range(len(self.TARGET_COLS))]

        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self.device)
                outputs = self.model(X_batch)

                for i, output in enumerate(outputs):
                    preds = output.argmax(dim=1).cpu().numpy()
                    all_predictions[i].extend(preds)

        predictions = np.column_stack(all_predictions)
        return predictions.astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """Predict probabilities for each target."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self.model.eval()
        loader = self._create_dataloader(X, None, shuffle=False)

        all_probas = [[] for _ in range(len(self.TARGET_COLS))]

        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self.device)
                outputs = self.model(X_batch)

                for i, output in enumerate(outputs):
                    proba = torch.softmax(output, dim=1).cpu().numpy()
                    all_probas[i].append(proba)

        return [np.vstack(p) for p in all_probas]
