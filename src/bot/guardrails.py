from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

class GuardrailsManager:
    def __init__(self):
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, 
            supported_languages=["en"]
        )
        
        plate_pattern = Pattern(
            name="license_plate_pattern",
            regex=r"\b[A-Z]{2,3}[0-9]{4,5}\b",
            score=0.85
        )
        plate_recognizer = PatternRecognizer(
            supported_entity="LICENSE_PLATE",
            patterns=[plate_pattern]
        )
        self.analyzer.registry.add_recognizer(plate_recognizer)

        phone_pattern = Pattern(
            name="custom_phone_pattern",
            regex=r"\+?\b\d{11,11}\b",
            score=0.85
        )

        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[phone_pattern]
        )
        self.analyzer.registry.add_recognizer(phone_recognizer)
        
        
        self.vault: dict[str, str] = {}
        self._counter = 0

    def tokenize(self, text: str) -> str:
        results = self.analyzer.analyze(
            text=text,
            entities=["PHONE_NUMBER", "PERSON", "LICENSE_PLATE"],
            language='en',
            score_threshold=0.6
        )
        
        valid_results = []
        for res in results:
            raw_val = text[res.start:res.end].strip()
            val_lower = raw_val.lower()
            
            if res.entity_type == "LICENSE_PLATE":
                if not any(char.isdigit() for char in raw_val):
                    continue
            
            if res.entity_type == "PERSON":
                if len(raw_val) < 2:
                    continue
                ignore_words = ["hi", "hello", "hey", "price", "parking", "sector", "please", "want", "check", "spot", "spots", "available"]
                if val_lower in ignore_words:
                    continue
                if "sector" in val_lower:
                    continue

            valid_results.append(res)
        
        valid_results.sort(key=lambda x: x.start, reverse=True)
        
        tokenized_text = text
        for res in valid_results:
            raw_value = tokenized_text[res.start:res.end]
            token_key = f"[TOKEN_{res.entity_type}_{self._counter}]"
            
            print(f"[GUARDRAILS DEBUG] Enum: '{res.entity_type}' | Result: {res.score} | Value: '{raw_value}'")
            
            self.vault[token_key] = raw_value
            tokenized_text = tokenized_text[:res.start] + token_key + tokenized_text[res.end:]
            self._counter += 1
            
        return tokenized_text

    def detokenize(self, data):
        """
        Recovers real data from tokens.
        Accepts a dictionary (ex. data from reservation) or plain text.
        """
        if isinstance(data, dict):
            clean_dict = {}
            for key, val in data.items():
                if isinstance(val, str):
                    new_val = val
                    for token, real_value in self.vault.items():
                        new_val = new_val.replace(token, real_value)
                    clean_dict[key] = new_val
                else:
                    clean_dict[key] = val
            return clean_dict
            
        elif isinstance(data, str):
            new_str = data
            for token, real_value in self.vault.items():
                new_str = new_str.replace(token, real_value)
            return new_str
            
        return data

    def clear_vault(self):
        """
        Cleans the secure vault after completing a transaction.
        """
        self.vault.clear()
        self._counter = 0

guardrails = GuardrailsManager()