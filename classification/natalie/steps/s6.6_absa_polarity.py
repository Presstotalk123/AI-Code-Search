import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)

_ABSA_TOKENIZER = None
_ABSA_MODEL = None
_DOC_SENTIMENT_PIPE = None
_DEVICE = None

# --- CONFIGURATION ---
INPUT_FILE = "sruthi_output.jsonl"
OUTPUT_FILE = "dhruv_output.jsonl"

def _resolve_devices(device_preference: str = "auto") -> tuple[torch.device, int]:
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), 0
    if device_preference == "cpu":
        return torch.device("cpu"), -1
    if torch.cuda.is_available():
        return torch.device("cuda"), 0
    return torch.device("cpu"), -1


def _load_models(device_preference: str = "auto"):
    global _ABSA_TOKENIZER
    global _ABSA_MODEL
    global _DOC_SENTIMENT_PIPE
    global _DEVICE

    if (
        _ABSA_TOKENIZER is not None
        and _ABSA_MODEL is not None
        and _DOC_SENTIMENT_PIPE is not None
        and _DEVICE is not None
    ):
        return _ABSA_TOKENIZER, _ABSA_MODEL, _DOC_SENTIMENT_PIPE, _DEVICE

    print("Loading Person 3 Models (ABSA & Sentiment)...")
    device, pipeline_device = _resolve_devices(device_preference=device_preference)

    # ABSA Model
    absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1", use_fast=False)
    absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    absa_model.to(device)

    # Fallback Sentiment Model
    doc_sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=pipeline_device,
    )

    _ABSA_TOKENIZER = absa_tokenizer
    _ABSA_MODEL = absa_model
    _DOC_SENTIMENT_PIPE = doc_sentiment_pipe
    _DEVICE = device
    logger.info("s6.6 models loaded (device=%s)", str(device))

    return _ABSA_TOKENIZER, _ABSA_MODEL, _DOC_SENTIMENT_PIPE, _DEVICE

def get_absa_label(text, aspect):
    absa_tokenizer, absa_model, _doc_sentiment_pipe, device = _load_models(device_preference="auto")
    inputs = absa_tokenizer(
    text,
    aspect,
    return_tensors="pt",
    truncation=True,
    max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = absa_model(**inputs)
    label_id = torch.argmax(outputs.logits, dim=1).item()
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return mapping[label_id]


def _normalise_sentiment_label(label: str) -> str:
    lowered = (label or "").lower()
    if "positive" in lowered:
        return "positive"
    if "negative" in lowered:
        return "negative"
    return "neutral"

def derive_document_polarity(aspect_values):
    unique_values = set(aspect_values)
    if len(unique_values) == 1:
        return list(unique_values)[0]
    return "mixed"


def run(records: list[dict], *, scores_buffer: list[dict] | None = None, device_preference: str = "auto") -> list[dict]:
    absa_tokenizer, absa_model, doc_sentiment_pipe, device = _load_models(device_preference=device_preference)
    errors = 0

    for record in records:
        clean_text = (record.get("clean_text") or "").strip()
        labels = record.setdefault("labels", {})
        subjectivity = labels.get("subjectivity")
        aspects = labels.get("aspects")

        try:
            if not clean_text:
                labels["polarity"] = "neutral"
                if isinstance(aspects, dict):
                    for key in list(aspects):
                        aspects[key] = "neutral"
            elif subjectivity == "neutral":
                labels["polarity"] = "neutral"
                if isinstance(aspects, dict):
                    for key in list(aspects):
                        aspects[key] = "neutral"
            else:
                if not isinstance(aspects, dict) or not aspects:
                    result = doc_sentiment_pipe(clean_text, truncation=True, max_length=512)[0]
                    labels["polarity"] = _normalise_sentiment_label(str(result.get("label", "neutral")))
                else:
                    aspect_results = []
                    for aspect_term in list(aspects.keys()):
                        inputs = absa_tokenizer(
                            clean_text,
                            aspect_term,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = absa_model(**inputs)
                        label_id = torch.argmax(outputs.logits, dim=1).item()
                        mapping = {0: "negative", 1: "neutral", 2: "positive"}
                        sentiment = mapping[label_id]
                        aspects[aspect_term] = sentiment
                        aspect_results.append(sentiment)

                    labels["polarity"] = derive_document_polarity(aspect_results)

            if scores_buffer is not None:
                scores_buffer.append(
                    {
                        "doc_id": record.get("doc_id"),
                        "polarity": labels.get("polarity"),
                        "aspect_polarity": aspects if isinstance(aspects, dict) else {},
                    }
                )
        except Exception as e:
            errors += 1
            logger.warning("s6.6 failed for doc_id=%s: %s", record.get("doc_id"), e)
            labels["polarity"] = "neutral"

    logger.info("s6.6 complete: %d records, %d errors", len(records), errors)
    return records

def process_sentiment_analysis():
    print(f"Processing records for Person 3 task...")
    count = 0
    
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if not line.strip():
                continue
            records.append(json.loads(line))

    records = run(records, device_preference="auto")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for record in records:
            f_out.write(json.dumps(record) + '\n')
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} records...")

    print(f"Finished! Person 3 output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_sentiment_analysis()