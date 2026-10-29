from presidio_analyzer import AnalyzerEngine
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
        
        self.vault: dict[str, str] = {}
        self._counter = 0

    def tokenize(self, text: str) -> str:
        """
        Replaces sensitive data (ex. phone number, name) by tokens,
        and true values saves in vault.
        """
        results = self.analyzer.analyze(
            text=text,
            entities=["PHONE_NUMBER", "PERSON"],
            language='en'
        )
        
        results.sort(key=lambda x: x.start, reverse=True)
        
        tokenized_text = text
        for res in results:
            raw_value = tokenized_text[res.start:res.end]
            token_key = f"[TOKEN_{res.entity_type}_{self._counter}]"
            
            self.vault[token_key] = raw_value
            
            tokenized_text = tokenized_text[:res.start] + token_key + tokenized_text[res.end:]
            self._counter += 1
            
        return tokenized_text

    def detokenize(self, data):
        """
        Recovers real data from tokens.
        Accepts a dictionary (ex. data from rezervation) or plain text.
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

guardrails = GuardrailsManager()