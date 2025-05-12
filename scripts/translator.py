# translator.py
from transformers import MarianMTModel, MarianTokenizer
import torch
import argparse
import json
import numpy as np

def translate_with_confidence(text, model_name="/home/maquiroz/mixtec_translation_project/models/final_model"):
    try:
        # Load the model and tokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        # Generate the translation with output scores
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                output_attentions=True,
                max_length=128
            )
        
        # Get the generated token IDs
        token_ids = outputs.sequences[0].tolist()
        
        # Get the scores for each token
        scores = torch.stack(outputs.scores, dim=1)
        
        # Get the log probabilities by applying softmax to the scores
        log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
        
        # For each position, get the log probability of the chosen token
        token_log_probs = [
            float(log_probs[0, i, token_id].item())
            for i, token_id in enumerate(token_ids[1:])  # Skip the BOS token
        ]
        
        # Convert log probabilities to confidence scores (0-100%)
        confidence_scores = [min(100, max(0, round((1 - np.exp(score)) * 100))) for score in token_log_probs]
        
        # Decode the translation
        translated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Get word alignments by splitting text into tokens
        translated_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        translated_tokens = [t for t in translated_tokens if t not in tokenizer.all_special_tokens]
        
        # Clean up tokens that use subword tokenization (starting with ##)
        words = []
        confidence = []
        current_word = ""
        current_confidence = 0
        word_count = 0
        
        for token, score in zip(translated_tokens, confidence_scores):
            if token.startswith("##"):
                current_word += token[2:]
                # Average the confidence scores for subwords
                current_confidence = (current_confidence * word_count + score) / (word_count + 1)
                word_count += 1
            else:
                if current_word:
                    words.append(current_word)
                    confidence.append(round(current_confidence))
                current_word = token
                current_confidence = score
                word_count = 1
        
        # Add the last word
        if current_word:
            words.append(current_word)
            confidence.append(round(current_confidence))
        
        # Create a list of word/confidence pairs
        word_confidence = [{"word": w, "confidence": c} for w, c in zip(words, confidence)]
        
        # Calculate overall confidence
        overall_confidence = round(sum(confidence) / len(confidence)) if confidence else 0
        
        return {
            "success": True, 
            "translation": translated_text,
            "word_confidence": word_confidence,
            "overall_confidence": overall_confidence
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Spanish to Arabic with confidence scores")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--model", default="/home/maquiroz/mixtec_translation_project/models/final_model", help="Model path")
    
    args = parser.parse_args()
    
    result = translate_with_confidence(args.text, args.model)
    print(json.dumps(result))