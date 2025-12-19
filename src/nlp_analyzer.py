"""NLP модуль для семантического анализа медицинских текстов."""
from typing import List, Tuple, Optional, Dict, Any
import re
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

from .data_preprocessor import PatientData, Criterion


def _find_valid_snapshot(snapshots_dir: Path) -> Optional[Path]:
    """Find the snapshot directory that has a valid config.json with model_type."""
    import json
    if not snapshots_dir.exists():
        return None

    for snap in snapshots_dir.iterdir():
        if not snap.is_dir():
            continue
        config_path = snap / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if "model_type" in config:
                    # Also check if it has model weights
                    has_weights = any(
                        (snap / name).exists()
                        for name in ["pytorch_model.bin", "model.safetensors"]
                    )
                    if has_weights:
                        return snap
            except:
                pass

    # Fallback: just find one with model weights
    for snap in snapshots_dir.iterdir():
        if not snap.is_dir():
            continue
        has_weights = any(
            (snap / name).exists()
            for name in ["pytorch_model.bin", "model.safetensors"]
        )
        if has_weights:
            return snap

    return None


def get_model_path() -> str:
    """
    Возвращает путь к модели SapBERT.
    SapBERT - медицинская модель (cambridgeltl/SapBERT-from-PubMedBERT-fulltext).
    """
    model_dir = "models--cambridgeltl--SapBERT-from-PubMedBERT-fulltext"

    # Docker path
    docker_cache = Path(f"/root/.cache/huggingface/hub/{model_dir}/snapshots")
    if docker_cache.exists():
        snapshot = _find_valid_snapshot(docker_cache)
        if snapshot:
            return str(snapshot)

    # Local path (Windows)
    local_model = Path(__file__).parent.parent / "models" / model_dir / "snapshots"
    if local_model.exists():
        snapshot = _find_valid_snapshot(local_model)
        if snapshot:
            return str(snapshot)

    # Fallback - download from internet
    return "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class MedicalNLPAnalyzer:
    """
    Анализатор медицинских текстов на основе sentence-transformers.
    """

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = get_model_path()
        print(f"Loading SapBERT model from: {model_name}")
        self.model = SentenceTransformer(model_name)
        # Пороги для определения
        self.match_threshold = 0.45  # Порог для совпадения
        self.high_confidence = 0.60  # Высокая уверенность
        self.exclusion_threshold = 0.65  # Порог для exclusion (с affirmation check)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Кодирует список текстов в эмбеддинги."""
        if not texts:
            return torch.tensor([])
        return self.model.encode(texts, convert_to_tensor=True)

    def find_best_match(
        self,
        criterion: str,
        sentences: List[str],
        sentence_embeddings: Optional[torch.Tensor] = None,
        use_expansion: bool = True
    ) -> Tuple[str, float, int]:
        """
        Находит лучшее совпадение для критерия среди предложений.

        Returns:
            (best_sentence, score, index)
        """
        if not sentences:
            return ("", 0.0, -1)

        # Расширяем критерий медицинскими синонимами для лучшего matching
        if use_expansion:
            criterion_expanded = self._expand_medical_terms(criterion)
        else:
            criterion_expanded = criterion

        criterion_emb = self.model.encode(criterion_expanded, convert_to_tensor=True)

        # Для предложений также применяем расширение если они не закешированы
        if sentence_embeddings is None:
            if use_expansion:
                expanded_sentences = [self._expand_medical_terms(s) for s in sentences]
                sentence_embeddings = self.encode_texts(expanded_sentences)
            else:
                sentence_embeddings = self.encode_texts(sentences)

        similarities = util.cos_sim(criterion_emb, sentence_embeddings)[0]
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[best_idx].item()

        return (sentences[best_idx], best_score, best_idx)

    def _expand_medical_terms(self, text: str) -> str:
        """
        Расширяет медицинские термины синонимами для лучшего matching.
        """
        # Маппинг синонимов
        synonyms = {
            # SGLT2 inhibitors
            'empagliflozin': 'empagliflozin sglt2 inhibitor jardiance',
            'dapagliflozin': 'dapagliflozin sglt2 inhibitor farxiga forxiga',
            'canagliflozin': 'canagliflozin sglt2 inhibitor invokana',
            'ertugliflozin': 'ertugliflozin sglt2 inhibitor steglatro',
            'sglt2': 'sglt2 inhibitor gliflozin empagliflozin dapagliflozin canagliflozin',
            'gliflozin': 'sglt2 inhibitor gliflozin',
            # Heart conditions
            'hfref': 'hfref heart failure reduced ejection fraction systolic',
            'hfpef': 'hfpef heart failure preserved ejection fraction diastolic',
            'ef': 'ef ejection fraction lvef',
            'lvef': 'lvef ef left ventricular ejection fraction',
            # Diabetes
            't1dm': 't1dm type 1 diabetes mellitus',
            't2dm': 't2dm type 2 diabetes mellitus',
            # Cardiovascular events
            'mi': 'mi myocardial infarction heart attack',
            'tia': 'tia transient ischemic attack mini stroke',
            'crt': 'crt cardiac resynchronization therapy',
            'icd': 'icd implantable cardioverter defibrillator',
            # Blood pressure
            'bp': 'bp blood pressure',
            'sbp': 'sbp systolic blood pressure',
            'dbp': 'dbp diastolic blood pressure',
        }

        text_lower = text.lower()
        expanded = text_lower

        for term, expansion in synonyms.items():
            if term in text_lower:
                expanded += ' ' + expansion

        return expanded

    def _check_affirmation(self, sentence: str, criterion_text: str) -> bool:
        """
        Проверяет наличие утвердительных паттернов в предложении.
        Для exclusion критериев требуется явное подтверждение наличия состояния.
        """
        sentence_lower = sentence.lower()
        criterion_lower = criterion_text.lower()

        # Извлекаем ключевые слова из критерия
        criterion_keywords = []
        medical_terms = [
            'diabetes', 'hypotension', 'hypertension', 'transplant', 'infarction',
            'mi', 'stroke', 'tia', 'sglt2', 'inhibitor', 'angina', 'decompensated',
            'acute', 'pregnant', 'dialysis', 'cardiomyopathy', 'myocarditis',
            'pericarditis', 'arrhythmia', 'bradycardia', 'block', 'pacemaker',
            'icd', 'crt', 'valve', 'valvular', 'restrictive', 'constrictive',
            'hypertrophic', 'renal', 'hepatic', 'liver', 'kidney'
        ]
        for term in medical_terms:
            if term in criterion_lower:
                criterion_keywords.append(term)

        # Проверяем наличие ключевых слов в предложении
        has_keyword_match = any(kw in sentence_lower for kw in criterion_keywords)
        if not has_keyword_match:
            # Проверяем частичные совпадения (напр. "cardiomyopathy" в предложении содержит "cardio")
            criterion_words = [w for w in re.findall(r'\w+', criterion_lower) if len(w) > 4]
            has_keyword_match = any(cw in sentence_lower for cw in criterion_words)

        if not has_keyword_match:
            return False

        # Паттерны явного утверждения наличия состояния
        affirmation_patterns = [
            r'\bhas\s+(?:a\s+)?(?:history\s+of\s+)?',
            r'\bwith\s+(?:known\s+)?(?:a\s+)?(?:history\s+of\s+)?',
            r'\bdiagnosed\s+with\b',
            r'\bsuffers?\s+from\b',
            r'\bpresents?\s+with\b',
            r'\breceiving\s+(?:therapy\s+with\s+)?',
            r'\btaking\b',
            r'\bon\s+(?:therapy\s+(?:with|for)\s+)?',
            r'\bcurrently\s+(?:on|taking|receiving)\b',
            r'\brecent(?:ly)?\b',
            r'\bacute\b',
            r'\bactive\b',
            r'\bknown\b',
            r'\bconfirmed\b',
            r'\bdocumented\b',
            r'\bprior\b',
            r'\bprevious\b',
            r'\bhistory\s+of\b',
            r'\bsecondary\s+to\b',
            r'\bdue\s+to\b',
            r'\bcaused\s+by\b',
        ]

        # Проверяем наличие утвердительного паттерна
        for pattern in affirmation_patterns:
            if re.search(pattern, sentence_lower):
                return True

        return False

    def _check_direct_exclusion_affirmation(self, criterion_text: str, note_sentences: List[str]) -> bool:
        """
        Проверяет прямое утверждение для exclusion критериев.
        Если найдено — возвращает True, и критерий должен быть MET.
        """
        criterion_lower = criterion_text.lower()

        # Маппинг критериев к паттернам прямого утверждения
        direct_affirmation_patterns = {
            'type 1 diabetes': [
                r'type\s+1\s+diabetes\s+(?:since|for|diagnosed|mellitus|on\s+insulin|managed)',
                r'has\s+(?:had\s+)?type\s+1\s+diabetes',
                r'with\s+type\s+1\s+diabetes',
                r't1dm\s+(?:since|for|on\s+insulin)',
            ],
            'sglt2 inhibitor': [
                r'(?:on|taking|receiving|started)\s+(?:\w+\s+)*?(?:empagliflozin|dapagliflozin|canagliflozin|ertugliflozin)',
                r'(?:on|taking|receiving|started)\s+(?:\w+\s+)*?sglt2',
                r'(?:on|taking|receiving|started)\s+(?:\w+\s+)*?(?:jardiance|farxiga|invokana)',
                r'current(?:ly)?\s+(?:on|taking)\s+(?:\w+\s+)*?(?:sglt2|gliflozin)',
            ],
            'cardiac transplant': [
                r'(?:cardiac|heart)\s+transplant\s+(?:recipient|patient|in\s+\d)',
                r'post[- ](?:cardiac|heart)\s+transplant',
                r's/p\s+(?:cardiac|heart)\s+transplant',
            ],
            'dialysis': [
                r'(?:on|receiving)\s+(?:chronic\s+)?(?:hemo)?dialysis',
                r'esrd\s+on\s+dialysis',
            ],
            'cardiovascular event': [
                r'recent\s+cardiovascular\s+event',
                r'recent\s+(?:major\s+)?(?:cardiac|cardiovascular)\s+event',
                r'recent\s+(?:mi|myocardial\s+infarction|stroke)',
            ],
            'statin intolerance': [
                r'statin\s+intolerance',
                r'intolerance\s+(?:to\s+)?statin',
                r'prior\s+statin\s+intolerance',
                r'documented\s+statin\s+intolerance',
            ],
            'pregnant': [
                r'\bpatient\s+is\s+pregnant\b',
                r'\bpregnancy\s+noted\b',
                r'\bcurrently\s+pregnant\b',
            ],
        }

        for criterion_key, patterns in direct_affirmation_patterns.items():
            if criterion_key in criterion_lower:
                for sent in note_sentences:
                    sent_lower = sent.lower()
                    for pattern in patterns:
                        if re.search(pattern, sent_lower):
                            # Проверяем, нет ли отрицания в этом конкретном предложении
                            negation_before = re.search(r'\b(?:no|not|without|denies?|never)\b.*' + pattern, sent_lower)
                            if not negation_before:
                                return True
        return False

    def _check_keyword_match(self, criterion_text: str, note_sentences: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Проверяет наличие ключевых слов критерия в предложениях.
        Возвращает (has_match, matched_sentence).
        """
        criterion_lower = criterion_text.lower()

        # Специфичные exclusion паттерны с высокой точностью
        exclusion_keywords = {
            'type 1 diabetes': ['type 1 diabetes', 't1dm', 'type i diabetes'],
            'symptomatic hypotension': ['symptomatic hypotension', 'orthostatic hypotension'],
            'sglt2 inhibitor': ['sglt2', 'sglt-2', 'empagliflozin', 'dapagliflozin', 'canagliflozin', 'jardiance', 'farxiga'],
            'cardiac transplant': ['cardiac transplant', 'heart transplant', 'transplant list'],
            'dialysis': ['dialysis', 'hemodialysis', 'peritoneal dialysis', 'esrd'],
            'pregnant': ['pregnant', 'pregnancy'],
            'acute coronary': ['acute coronary', 'acute mi', 'acute myocardial', 'stemi', 'nstemi'],
            'stroke': ['stroke', 'cva', 'cerebrovascular'],
            'restrictive cardiomyopathy': ['restrictive cardiomyopathy'],
            'constrictive pericarditis': ['constrictive pericarditis'],
            'hypertrophic cardiomyopathy': ['hypertrophic cardiomyopathy', 'hocm'],
        }

        # Проверяем специфичные паттерны
        for pattern, keywords in exclusion_keywords.items():
            if pattern in criterion_lower:
                for sent in note_sentences:
                    sent_lower = sent.lower()
                    for kw in keywords:
                        if kw in sent_lower:
                            return (True, sent)

        # Общий keyword matching
        criterion_words = [w for w in re.findall(r'\w+', criterion_lower) if len(w) > 5]
        for sent in note_sentences:
            sent_lower = sent.lower()
            matches = sum(1 for w in criterion_words if w in sent_lower)
            # Требуем минимум 2 совпадения или 50% слов
            if matches >= 2 or (len(criterion_words) > 0 and matches / len(criterion_words) >= 0.5):
                return (True, sent)

        return (False, None)

    def _check_numeric_criterion(self, criterion_text: str, note_sentences: List[str]) -> Tuple[Optional[bool], Optional[str]]:
        """
        Проверяет числовые критерии (LVEF, eGFR, возраст, BP).
        Возвращает:
          - (True, matched_sentence) если критерий выполнен
          - (False, matched_sentence) если критерий НЕ выполнен
          - (None, None) если не удалось определить
        """
        criterion_lower = criterion_text.lower()

        # LVEF критерий: "LVEF ≤ 40%" или "LVEF <= 40%"
        lvef_criterion = re.search(r'lvef\s*[≤<]=?\s*(\d+)', criterion_lower)
        if lvef_criterion:
            threshold = int(lvef_criterion.group(1))
            # Ищем LVEF в предложениях
            for sentence in note_sentences:
                sentence_lower = sentence.lower()
                # Паттерны: "LVEF of 45%", "ejection fraction of 45%", "EF 45%"
                lvef_match = re.search(
                    r'(?:lvef|left\s+ventricular\s+ejection\s+fraction|ejection\s+fraction|ef)\s*(?:of\s+|is\s+|was\s+|:?\s*)(\d+)\s*%?',
                    sentence_lower
                )
                if lvef_match:
                    actual_lvef = int(lvef_match.group(1))
                    # Для критерия LVEF <= 40%: если actual > 40, критерий НЕ выполнен
                    if actual_lvef > threshold:
                        return (False, sentence)
                    else:
                        return (True, sentence)

        # eGFR критерий: "eGFR >= 30" (inclusion) или "eGFR < 30" (exclusion)
        egfr_ge_criterion = re.search(r'egfr\s*[≥>]=?\s*(\d+)', criterion_lower)
        egfr_lt_criterion = re.search(r'egfr\s*<\s*(\d+)', criterion_lower)

        if egfr_ge_criterion or egfr_lt_criterion:
            # Ищем eGFR в предложениях
            for sentence in note_sentences:
                sentence_lower = sentence.lower()
                egfr_match = re.search(r'egfr[^0-9]*(\d+)', sentence_lower)
                if egfr_match:
                    actual_egfr = int(egfr_match.group(1))
                    if egfr_ge_criterion:
                        threshold = int(egfr_ge_criterion.group(1))
                        # Для eGFR >= 30: если actual >= 30, критерий выполнен
                        return (actual_egfr >= threshold, sentence)
                    if egfr_lt_criterion:
                        threshold = int(egfr_lt_criterion.group(1))
                        # Для eGFR < 30: если actual < 30, критерий выполнен
                        return (actual_egfr < threshold, sentence)

        # Возраст: "aged >= 18" или "age >= 18" или "Age 35-80 years"
        # Сначала проверяем диапазон (Age 35-80)
        age_range = re.search(r'age\s*(\d+)\s*[-–—]\s*(\d+)\s*years?', criterion_lower)
        if age_range:
            min_age = int(age_range.group(1))
            max_age = int(age_range.group(2))
            for sentence in note_sentences:
                sentence_lower = sentence.lower()
                age_match = re.search(r'(\d+)[\s-]*year[\s-]*old', sentence_lower)
                if age_match:
                    actual_age = int(age_match.group(1))
                    # Для диапазона 35-80: если actual < 35 или > 80, критерий НЕ выполнен
                    if min_age <= actual_age <= max_age:
                        return (True, sentence)
                    else:
                        return (False, sentence)

        # Простой возраст: "aged >= 18"
        age_criterion = re.search(r'age[d]?\s*[≥>]=?\s*(\d+)', criterion_lower)
        if age_criterion:
            threshold = int(age_criterion.group(1))
            for sentence in note_sentences:
                sentence_lower = sentence.lower()
                age_match = re.search(r'(\d+)[\s-]*year[\s-]*old', sentence_lower)
                if age_match:
                    actual_age = int(age_match.group(1))
                    return (actual_age >= threshold, sentence)

        # BP критерий: "systolic BP < 95" или "BP <95"
        bp_criterion = re.search(r'(?:systolic\s+)?bp\s*<\s*(\d+)', criterion_lower)
        if bp_criterion:
            threshold = int(bp_criterion.group(1))
            for sentence in note_sentences:
                sentence_lower = sentence.lower()
                # Паттерн 1: BP 120/80
                bp_match = re.search(r'(?:bp|blood\s+pressure)[^0-9]*(\d+)/(\d+)', sentence_lower)
                if bp_match:
                    systolic = int(bp_match.group(1))
                    if systolic < threshold:
                        return (True, sentence)
                # Паттерн 2: SBP 93 или SBP 94 mmHg
                sbp_match = re.search(r'\bsbp\s*(?:of\s+|is\s+|:)?\s*(\d+)', sentence_lower)
                if sbp_match:
                    systolic = int(sbp_match.group(1))
                    if systolic < threshold:
                        return (True, sentence)
                # Паттерн 3: systolic 93 или systolic BP 93
                systolic_match = re.search(r'systolic\s*(?:bp\s*)?(?:of\s+|is\s+|:)?\s*(\d+)', sentence_lower)
                if systolic_match:
                    systolic = int(systolic_match.group(1))
                    if systolic < threshold:
                        return (True, sentence)

        # ALT/AST критерий: "ALT > 3× ULN" или "hepatic impairment"
        # ULN для ALT = 56 U/L, для AST = 40 U/L
        # 3× означает ALT > 168 или AST > 120
        if 'hepatic' in criterion_lower or 'alt' in criterion_lower or 'ast' in criterion_lower:
            if '3' in criterion_lower and 'uln' in criterion_lower:
                alt_threshold = 168  # 3 × 56
                ast_threshold = 120  # 3 × 40
                for sentence in note_sentences:
                    sentence_lower = sentence.lower()
                    # Ищем ALT
                    alt_match = re.search(r'\balt[:\s]*(\d+)', sentence_lower)
                    if alt_match:
                        actual_alt = int(alt_match.group(1))
                        if actual_alt > alt_threshold:
                            return (True, sentence)
                    # Ищем AST
                    ast_match = re.search(r'\bast[:\s]*(\d+)', sentence_lower)
                    if ast_match:
                        actual_ast = int(ast_match.group(1))
                        if actual_ast > ast_threshold:
                            return (True, sentence)
                # Если нашли ALT/AST, но они в норме — критерий не выполнен
                for sentence in note_sentences:
                    sentence_lower = sentence.lower()
                    if re.search(r'\balt[:\s]*\d+', sentence_lower) or re.search(r'\bast[:\s]*\d+', sentence_lower):
                        return (False, sentence)

        return (None, None)

    def _check_statin_initiation(self, note_sentences: List[str]) -> Tuple[Optional[bool], Optional[str]]:
        """
        Проверяет критерий "New initiation of statin" для Study W02.

        Паттерны, указывающие на НАЧАЛО приёма статина:
        - "start Atorvastatin 20 mg"
        - "initiate statin therapy"
        - "begin rosuvastatin"
        - "prescribed atorvastatin"

        "No prior statin use" — это ПОДТВЕРЖДЕНИЕ критерия (statin-naive = можно начать)
        """
        statin_names = [
            'atorvastatin', 'rosuvastatin', 'simvastatin', 'pravastatin',
            'lovastatin', 'fluvastatin', 'pitavastatin', 'statin'
        ]

        # Паттерны начала приёма статина
        initiation_patterns = [
            r'\b(?:start|started|starting|begin|began|initiat\w+|prescrib\w+|order\w+)\s+(?:\w+\s+)*?(' + '|'.join(statin_names) + r')',
            r'\bplan[:\s]+(?:start|initiat\w+|begin)\s+(?:\w+\s+)*?(' + '|'.join(statin_names) + r')',
            r'\b(' + '|'.join(statin_names) + r')\s+(?:\d+\s*mg\s+)?(?:daily|nightly|qd|qhs)\b.*(?:start|begin|initiat)',
        ]

        for sentence in note_sentences:
            sentence_lower = sentence.lower()
            for pattern in initiation_patterns:
                if re.search(pattern, sentence_lower):
                    return (True, sentence)

        # Также проверяем: если есть "no prior statin" И есть назначение статина в плане
        has_no_prior = False
        has_statin_in_plan = False
        statin_sentence = None

        for sentence in note_sentences:
            sentence_lower = sentence.lower()
            if 'no prior statin' in sentence_lower or 'no statin' in sentence_lower:
                has_no_prior = True
            # Проверяем назначение в плане
            if any(statin in sentence_lower for statin in statin_names[:7]):  # Конкретные статины
                if any(word in sentence_lower for word in ['plan', 'start', 'begin', 'prescrib', 'order', 'initiat']):
                    has_statin_in_plan = True
                    statin_sentence = sentence

        if has_no_prior and has_statin_in_plan:
            return (True, statin_sentence)

        return (None, None)

    def check_negation_in_context(self, sentence: str, criterion_keywords: List[str], criterion_text: str = "") -> bool:
        """
        Проверяет наличие отрицания в контексте ключевых слов критерия.
        ВАЖНО: negation должен предшествовать keyword, а не просто быть в том же предложении.
        """
        sentence_lower = sentence.lower()
        criterion_lower = criterion_text.lower()

        # Извлекаем ключевые медицинские термины из критерия
        criterion_terms = set()
        medical_terms = [
            'diabetes', 'dm', 'hypotension', 'hypertension', 'hf', 'heart failure',
            'sglt2', 'inhibitor', 'transplant', 'transplantation', 'infarction', 'mi', 'stroke', 'tia',
            'angina', 'arrhythmia', 'cardiomyopathy', 'valve', 'valvular', 'myocarditis', 'pericarditis',
            'pacemaker', 'icd', 'crt', 'dialysis', 'renal', 'hepatic', 'pregnant',
            'decompensated', 'acute', 'unstable', 'hospitalization', 'revascularization',
            'restrictive', 'constrictive', 'hypertrophic', 'bradycardia', 'block',
            'egfr', 'gfr', 'lvef', 'ef',
            # Дополнительные термины для cardiovascular критериев
            'cardiovascular', 'cardiac', 'coronary', 'malignancy', 'cancer', 'steroid',
            'glucocorticoid', 'prednisone', 'trial', 'interventional', 'breastfeeding',
            'statin', 'ldl', 'ascvd', 'familial', 'hypercholesterolemia'
        ]
        for term in medical_terms:
            if term in criterion_lower:
                criterion_terms.add(term)

        # Паттерны явного отрицания (negation должен предшествовать keyword)
        explicit_negation_patterns = [
            r'\bno\s+(?:history\s+of\s+)?(?:\w+\s+)*?{keyword}',
            r'\bno\s+evidence\s+of\s+(?:\w+\s+)*?{keyword}',
            r'\bwithout\s+(?:any\s+)?(?:\w+\s+)*?{keyword}',
            r'\bdenies\s+(?:any\s+)?(?:\w+\s+)*?{keyword}',
            r'\bdenied\s+(?:any\s+)?(?:\w+\s+)*?{keyword}',
            r'\bnot\s+(?:\w+\s+)*?{keyword}',
            r'\bnever\s+(?:had\s+)?(?:\w+\s+)*?{keyword}',
            r'\babsence\s+of\s+(?:\w+\s+)*?{keyword}',
            r'\bfree\s+of\s+(?:\w+\s+)*?{keyword}',
            r'\brules?\s+out\s+(?:\w+\s+)*?{keyword}',
            r'\bruled\s+out\s+(?:\w+\s+)*?{keyword}',
        ]

        # Проверяем термины из критерия
        for term in criterion_terms:
            for pattern_template in explicit_negation_patterns:
                pattern = pattern_template.format(keyword=re.escape(term))
                if re.search(pattern, sentence_lower):
                    return True

        # Проверяем ключевые слова напрямую
        for kw in criterion_keywords:
            if len(kw) < 4:  # Пропускаем короткие ключевые слова
                continue
            for pattern_template in explicit_negation_patterns:
                pattern = pattern_template.format(keyword=re.escape(kw.lower()))
                if re.search(pattern, sentence_lower):
                    return True

        # Специфичные паттерны для полного отрицания (не требуют keyword)
        strong_negation_phrases = [
            r'\bno\s+type\s+1\s+diabetes\b',
            r'\bno\s+prior\s+sglt2\b',
            r'\bno\s+sglt2\s+inhibitor\b',
            r'\bno\s+restrictive\b',
            r'\bno\s+hypertrophic\b',
            r'\bno\s+constrictive\b',
            r'\bno\s+myocarditis\b',
            r'\bno\s+pericarditis\b',
            r'\bno\s+cardiac\s+transplant\b',
            r'\bno\s+heart\s+transplant\b',
            r'\bnot\s+(?:on|taking|receiving)\s+(?:\w+\s+)*?sglt2\b',
            # Cardiovascular event patterns
            r'\bno\s+(?:recent\s+)?cardiovascular\s+event\b',
            r'\bno\s+(?:recent\s+)?(?:major\s+)?(?:cardiac|cardiovascular)\s+event\b',
            r'\bno\s+(?:recent\s+)?(?:myocardial\s+)?infarction\b',
            r'\bno\s+(?:recent\s+)?stroke\b',
            r'\bno\s+(?:recent\s+)?(?:coronary\s+)?revascularization\b',
            # Malignancy patterns
            r'\bno\s+(?:active\s+)?malignancy\b',
            r'\bno\s+(?:active\s+)?cancer\b',
            # Steroid patterns
            r'\bno\s+(?:chronic\s+)?(?:systemic\s+)?steroids?\b',
            r'\bno\s+(?:chronic\s+)?glucocorticoid\b',
            # Pregnancy patterns
            r'\bnot\s+pregnant\b',
            r'\bno\s+pregnancy\b',
            r'\bnot\s+breastfeeding\b',
            # Trial patterns
            r'\bno\s+(?:recent\s+)?(?:interventional\s+)?(?:clinical\s+)?trial\b',
            r'\bnot\s+(?:currently\s+)?(?:in|enrolled\s+in)\s+(?:a\s+)?(?:clinical\s+)?trial\b',
            # Statin patterns (for Study W02) - осторожно с "no prior statin"
            # "no prior statin" = statin-naive = хорошо для "new initiation"
            # Только явные отказы от statin терапии
            r'\bdiscontinued\s+statin\b',
            r'\bstopped\s+statin\b',
            r'\bstatin\s+intolerance\b',
        ]
        for pattern in strong_negation_phrases:
            if re.search(pattern, sentence_lower):
                return True

        # ВАЖНО: Если в предложении явно утверждается наличие условия — это НЕ negation
        # Например: "Type 1 diabetes since adolescence" — это утверждение, не отрицание
        affirmation_phrases = [
            r'type\s+1\s+diabetes\s+(?:since|for|diagnosed|mellitus|on\s+insulin)',
            r'has\s+(?:had\s+)?type\s+1\s+diabetes',
            r'with\s+type\s+1\s+diabetes',
            r'history\s+of\s+type\s+1\s+diabetes',
            r'(?:on|taking|receiving)\s+(?:\w+\s+)*?sglt2',
            r'currently\s+(?:on|taking)\s+(?:\w+\s+)*?(?:empagliflozin|dapagliflozin|canagliflozin)',
        ]
        for pattern in affirmation_phrases:
            if re.search(pattern, sentence_lower):
                return False  # Явное утверждение — не negation

        # Числовые значения, которые указывают на норму
        # BP >= 95 означает отсутствие гипотензии
        if 'hypotension' in criterion_lower or 'bp' in criterion_lower:
            bp_match = re.search(r'\b(?:bp|blood\s+pressure)\s*(?:is\s+)?(\d+)/(\d+)', sentence_lower)
            if bp_match:
                systolic = int(bp_match.group(1))
                if systolic >= 95:
                    return True

        # eGFR >= порога означает отсутствие тяжёлой почечной недостаточности
        if 'egfr' in criterion_lower or 'gfr' in criterion_lower:
            # Критерий типа "eGFR < 30"
            threshold_match = re.search(r'egfr\s*<\s*(\d+)', criterion_lower)
            if threshold_match:
                threshold = int(threshold_match.group(1))
                # Ищем значение eGFR в предложении
                egfr_match = re.search(r'egfr[^0-9]*(\d+)', sentence_lower)
                if egfr_match:
                    actual_egfr = int(egfr_match.group(1))
                    if actual_egfr >= threshold:
                        return True  # eGFR выше порога — критерий НЕ выполнен

        # Контекстно-зависимые паттерны для острых состояний
        # ВАЖНО: эти паттерны должны применяться только если в предложении есть связанный keyword
        acute_keywords = ['acute', 'decompensated', 'unstable']
        hf_keywords = ['hf', 'heart failure', 'cardiac', 'hospitalization']
        if any(kw in criterion_lower for kw in acute_keywords):
            # Проверяем, есть ли в предложении HF-related keyword
            has_hf_keyword = any(kw in sentence_lower for kw in hf_keywords)
            if has_hf_keyword:
                stability_patterns = [
                    r'\bstable\b(?!.*\bunstable)',
                    r'\bchronic\b(?!.*(?:decompensated|acute))',
                    r'\bwell[- ]controlled\b',
                    r'\bcompensated\b',
                ]
                for pattern in stability_patterns:
                    if re.search(pattern, sentence_lower):
                        return True

        return False

    def analyze_criterion_against_note(
        self,
        criterion: Criterion,
        note_sentences: List[str],
        sentence_embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Анализирует один критерий против клинических заметок.

        Returns:
            {
                'criterion': str,
                'is_met': bool | None,
                'confidence': str,
                'score': float,
                'matched_sentence': str,
                'has_negation': bool,
                'explanation': str
            }
        """
        # Сначала проверяем числовые критерии — они имеют высший приоритет
        numeric_result, numeric_sentence = self._check_numeric_criterion(criterion.text, note_sentences)
        if numeric_result is not None:
            # Числовой критерий определён точно
            return {
                'criterion': criterion.text,
                'type': criterion.type,
                'is_met': numeric_result,
                'confidence': 'high',
                'score': 1.0 if numeric_result else 0.0,
                'matched_sentence': numeric_sentence or '[numeric comparison]',
                'has_negation': False,
                'explanation': f"Numeric criterion: {'MET' if numeric_result else 'NOT MET'}"
            }

        # Специальная проверка для statin initiation (Study W02)
        criterion_lower = criterion.text.lower()
        if 'statin' in criterion_lower and ('initiation' in criterion_lower or 'new' in criterion_lower):
            statin_result, statin_sentence = self._check_statin_initiation(note_sentences)
            if statin_result is not None:
                return {
                    'criterion': criterion.text,
                    'type': criterion.type,
                    'is_met': statin_result,
                    'confidence': 'high',
                    'score': 0.95 if statin_result else 0.0,
                    'matched_sentence': statin_sentence or '[statin initiation check]',
                    'has_negation': False,
                    'explanation': f"Statin initiation: {'MET' if statin_result else 'NOT MET'}"
                }

        # Sentence embeddings уже содержат расширенные термины
        best_sentence, score, idx = self.find_best_match(
            criterion.text, note_sentences, sentence_embeddings, use_expansion=False
        )
        # Но для критерия применяем расширение
        criterion_expanded = self._expand_medical_terms(criterion.text)
        criterion_emb = self.model.encode(criterion_expanded, convert_to_tensor=True)
        similarities = util.cos_sim(criterion_emb, sentence_embeddings)[0]
        best_idx = torch.argmax(similarities).item()
        score = similarities[best_idx].item()
        best_sentence = note_sentences[best_idx]

        # Определяем результат с разной логикой для inclusion и exclusion
        if criterion.type == 'exclusion':
            # Для exclusion критериев: баланс между точностью и полнотой
            has_affirmation = self._check_affirmation(best_sentence, criterion.text)

            # СНАЧАЛА проверяем прямое утверждение для специфичных exclusion критериев
            # Если есть прямое утверждение — критерий MET, независимо от negation в других предложениях
            direct_affirmation = self._check_direct_exclusion_affirmation(criterion.text, note_sentences)
            if direct_affirmation:
                return {
                    'criterion': criterion.text,
                    'type': criterion.type,
                    'is_met': True,
                    'confidence': 'high',
                    'score': score,
                    'matched_sentence': best_sentence,
                    'has_negation': False,
                    'explanation': f"Score: {score:.3f} (high) [DIRECT AFFIRMATION]"
                }

            # Для exclusion критериев: проверяем отрицание во ВСЕХ предложениях
            # Если хоть одно предложение явно отрицает критерий — критерий не выполнен
            has_negation = False
            for sent in note_sentences:
                if self.check_negation_in_context(sent, criterion.keywords, criterion.text):
                    has_negation = True
                    break

            if has_negation:
                # Явное отрицание - критерий точно не выполнен
                confidence = 'high' if score >= self.high_confidence else 'medium'
                is_met = False
            elif score >= 0.72:
                # Очень высокий score — не требуем affirmation
                confidence = 'high'
                is_met = True
            elif score >= self.exclusion_threshold and has_affirmation:
                # Высокий score + affirmation → met
                confidence = 'high'
                is_met = True
            elif score >= self.high_confidence and has_affirmation:
                # Средний score + affirmation → met
                confidence = 'medium'
                is_met = True
            else:
                # Без affirmation или низкий score → not met
                confidence = 'low'
                is_met = False
        else:
            # Для inclusion критериев: нужно ПОДТВЕРДИТЬ что критерий выполнен
            # Проверяем negation только в лучшем совпадении
            has_negation = self.check_negation_in_context(best_sentence, criterion.keywords, criterion.text)

            if score >= self.high_confidence:
                confidence = 'high'
                is_met = not has_negation
            elif score >= self.match_threshold:
                confidence = 'medium'
                is_met = not has_negation
            else:
                confidence = 'low'
                is_met = None  # Недостаточно данных для inclusion

        # Формируем объяснение
        neg_note = " [NEGATION DETECTED]" if has_negation else ""
        explanation = f"Score: {score:.3f} ({confidence}){neg_note}"

        return {
            'criterion': criterion.text,
            'type': criterion.type,
            'is_met': is_met,
            'confidence': confidence,
            'score': score,
            'matched_sentence': best_sentence,
            'has_negation': has_negation,
            'explanation': explanation
        }

    def evaluate_patient(self, patient: PatientData) -> Dict[str, Any]:
        """
        Полная оценка пациента по всем критериям.

        Returns:
            {
                'patient_id': str,
                'predicted_status': str,
                'ground_truth': str,
                'is_correct': bool,
                'inclusion_results': List[Dict],
                'exclusion_results': List[Dict],
                'summary': str
            }
        """
        # Кодируем все предложения из note с расширением медицинских терминов
        expanded_sentences = [self._expand_medical_terms(s) for s in patient.note_sentences]
        sentence_embeddings = self.encode_texts(expanded_sentences)

        inclusion_results = []
        exclusion_results = []

        # Анализируем критерии включения
        all_inclusion_met = True
        any_inclusion_unknown = False

        for criterion in patient.inclusion_criteria:
            result = self.analyze_criterion_against_note(
                criterion, patient.note_sentences, sentence_embeddings
            )
            inclusion_results.append(result)

            if result['is_met'] is False:
                all_inclusion_met = False
            elif result['is_met'] is None:
                any_inclusion_unknown = True

        # Анализируем критерии исключения
        any_exclusion_met = False
        any_exclusion_unknown = False

        for criterion in patient.exclusion_criteria:
            result = self.analyze_criterion_against_note(
                criterion, patient.note_sentences, sentence_embeddings
            )
            exclusion_results.append(result)

            if result['is_met'] is True:
                any_exclusion_met = True
            elif result['is_met'] is None:
                any_exclusion_unknown = True

        # Подсчитываем статистику по критериям
        inclusion_met_count = sum(1 for r in inclusion_results if r['is_met'] is True)
        inclusion_not_met_count = sum(1 for r in inclusion_results if r['is_met'] is False)
        inclusion_unknown_count = sum(1 for r in inclusion_results if r['is_met'] is None)
        total_inclusion = len(inclusion_results)

        exclusion_met_count = sum(1 for r in exclusion_results if r['is_met'] is True)
        exclusion_not_met_count = sum(1 for r in exclusion_results if r['is_met'] is False)
        exclusion_unknown_count = sum(1 for r in exclusion_results if r['is_met'] is None)
        total_exclusion = len(exclusion_results)

        # Собираем unknown критерии для детализации
        unknown_criteria = []
        for r in inclusion_results:
            if r['is_met'] is None:
                unknown_criteria.append(f"[INCLUSION] {r['criterion'][:50]}...")
        for r in exclusion_results:
            if r['is_met'] is None:
                unknown_criteria.append(f"[EXCLUSION] {r['criterion'][:50]}...")

        # Вычисляем процент unknown критериев
        total_criteria = total_inclusion + total_exclusion
        total_unknown = inclusion_unknown_count + exclusion_unknown_count
        unknown_ratio = total_unknown / total_criteria if total_criteria > 0 else 0
        UNKNOWN_THRESHOLD = 0.20  # Порог: >20% unknown → not enough info

        # Определяем статус — гибридная логика
        # 1. Если любой exclusion met → excluded (точно не подходит)
        # 2. Если любой inclusion NOT met → excluded (точно не подходит)
        # 3. Если >20% критериев unknown → not enough information (нужно уточнение)
        # 4. Если ≤20% unknown и есть met inclusion → included (решение на основе известных)
        if exclusion_met_count > 0:
            # Хотя бы один exclusion критерий выполнен — исключаем
            predicted_status = 'excluded'
            summary = f"EXCLUDED: {exclusion_met_count} exclusion criteria met"
        elif inclusion_not_met_count > 0:
            # Есть явно невыполненные inclusion критерии — исключаем
            predicted_status = 'excluded'
            summary = f"EXCLUDED: {inclusion_not_met_count} inclusion criteria not met"
        elif unknown_ratio > UNKNOWN_THRESHOLD:
            # Слишком много неопределённых критериев — нужно уточнение
            predicted_status = 'not enough information'
            unknown_details = "; ".join(unknown_criteria[:3])  # Показываем первые 3
            if len(unknown_criteria) > 3:
                unknown_details += f" (+{len(unknown_criteria) - 3} more)"
            summary = f"NEEDS CLARIFICATION: {total_unknown}/{total_criteria} ({unknown_ratio:.0%}) criteria unknown — {unknown_details}"
        elif inclusion_met_count > 0:
            # Достаточно известных критериев для решения — включаем
            if total_unknown > 0:
                predicted_status = 'included'
                summary = f"INCLUDED: {inclusion_met_count}/{total_inclusion} inclusion met, {total_unknown} unknown (≤{UNKNOWN_THRESHOLD:.0%})"
            else:
                predicted_status = 'included'
                summary = f"INCLUDED: {inclusion_met_count}/{total_inclusion} inclusion met, no exclusion met"
        else:
            # Нет информации вообще
            predicted_status = 'not enough information'
            summary = "NOT ENOUGH INFO: No criteria could be evaluated"

        # Сравниваем с ground truth
        is_correct = False
        if patient.ground_truth:
            gt_normalized = patient.ground_truth.lower()
            if 'include' in gt_normalized and predicted_status == 'included':
                is_correct = True
            elif 'exclude' in gt_normalized and predicted_status == 'excluded':
                is_correct = True
            elif ('not enough' in gt_normalized or 'info' in gt_normalized) and predicted_status == 'not enough information':
                is_correct = True

        return {
            'patient_id': patient.patient_id,
            'predicted_status': predicted_status,
            'ground_truth': patient.ground_truth or 'N/A',
            'is_correct': is_correct,
            'inclusion_results': inclusion_results,
            'exclusion_results': exclusion_results,
            'summary': summary
        }


# Глобальный экземпляр
_analyzer: Optional[MedicalNLPAnalyzer] = None


def get_analyzer() -> MedicalNLPAnalyzer:
    """Возвращает глобальный экземпляр анализатора."""
    global _analyzer
    if _analyzer is None:
        _analyzer = MedicalNLPAnalyzer()
    return _analyzer
