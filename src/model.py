from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

TOXIC_KEYWORDS = ("idiot", "stupid", "trash", "moron", "hate")


class ToxicityModel:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded = False
        self.model_name = "unitary/toxic-bert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self._is_loaded = True

    def score(self, text: str) -> float:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded yet.")
        assert self.tokenizer is not None
        assert self.model is not None
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits.sigmoid()[0, 0].item()
        return probs

    def predict(self, text: str) -> bool:
        return self.score(text) > 0.5
