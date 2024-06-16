import json


def process_docs(dataset):

    label2id= {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

    id2label = {v: k for k, v in label2id.items()}

    def _label_tokens(doc):
        doc['ner_labels'] = [id2label[x] for x in doc['ner_tags']]
        return doc
    
    def _tokens_to_sentence(doc):
        doc['sentence'] = ' '.join(doc['tokens'])
        return doc

    def _extract_entities(doc):
        entities = {'PER': [], 'ORG': [], 'LOC': [], 'MISC': []}
        current_entity = {"type": None, "tokens": []}
        for token, label in zip(doc['tokens'], doc['ner_labels']):
            if label.startswith('B-'):
                entity_type = label.split('-')[1]
                if current_entity["type"] == entity_type:
                    entities[entity_type].append(' '.join(current_entity["tokens"]))
                    current_entity["tokens"] = [token]
                else:
                    if current_entity["type"] is not None:
                        entities[current_entity["type"]].append(' '.join(current_entity["tokens"]))
                    current_entity = {"type": entity_type, "tokens": [token]}
            elif label.startswith('I-'):
                if current_entity["type"] is not None:
                    current_entity["tokens"].append(token)
            else:
                if current_entity["type"] is not None:
                    entities[current_entity["type"]].append(' '.join(current_entity["tokens"]))
                current_entity = {"type": None, "tokens": []}
        if current_entity["type"] is not None:
            entities[current_entity["type"]].append(' '.join(current_entity["tokens"]))

        doc['entities'] = entities
        return doc

    dataset=dataset.map(_label_tokens)
    dataset=dataset.map(_tokens_to_sentence)
    dataset=dataset.map(_extract_entities)

    return dataset



def is_valid_json(json_string):
    """
    Check if the given string is correctly formatted JSON.

    Args:
        json_string (str): The string to check.

    Returns:
        bool: True if the string is valid JSON, False otherwise.
    """
    try:
        json.loads(json_string)
        return True
    except:
        return False
    


def precision(actual, predicted):

    actual = list(set(actual))
    predicted = list(set(predicted))

    actual_lower = [word.lower() for word in actual]
    predicted_lower = [word.lower() for word in predicted]

    if not actual_lower and not predicted_lower:
        return 1.0  # Both lists are empty, so precision is 1 (correct prediction)

    true_positives = sum(1 for p in predicted_lower if p in actual_lower)
    predicted_positives = len(predicted_lower)
    if predicted_positives == 0:
        return 0  # Handle case where there are no predicted positives to avoid division by zero
    return true_positives / predicted_positives

def recall(actual, predicted):
    # remove duplicates
    actual = list(set(actual))
    predicted = list(set(predicted))

    #lower case
    actual_lower = [word.lower() for word in actual]
    predicted_lower = [word.lower() for word in predicted]

    if not actual_lower and not predicted_lower:
        return 1.0  # Both lists are empty, so recall is 1 (correct prediction)

    true_positives = sum(1 for p in predicted_lower if p in actual_lower)
    actual_positives = len(actual_lower)
    if actual_positives == 0:
        return 0  # Handle case where there are no actual positives to avoid division by zero
    return true_positives / actual_positives


def f1_score(actual, predicted):
    prec = precision(actual, predicted)
    rec = recall(actual, predicted)
    if prec + rec == 0:
        return prec, rec, 0  # Handle case where precision + recall is zero to avoid division by zero
    return prec, rec, 2 * (prec * rec) / (prec + rec)


def process_results(doc, results):
    # ground truth
    entities = doc['entities']
    
    # Initialize precision, recall, and F1 scores
    p1, r1, f1 = 0, 0, 0
    p2, r2, f2 = 0, 0, 0
    p3, r3, f3 = 0, 0, 0
    p4, r4, f4 = 0, 0, 0

    # check if response is parseable json
    js = is_valid_json(*results)
    if js:
        response_dict = json.loads(*results)
        # if response dict has PER key
        if 'PER' in response_dict:
            p1, r1, f1 = f1_score(entities['PER'], response_dict["PER"])
        if 'ORG' in response_dict:
            p2, r2, f2 = f1_score(entities['ORG'], response_dict["ORG"])
        if 'LOC' in response_dict:
            p3, r3, f3 = f1_score(entities['LOC'], response_dict["LOC"])
        if "MISC" in response_dict:
            p4, r4, f4 = f1_score(entities['MISC'], response_dict["MISC"])
    
    
    return {"is_json": js,
            "prec_PER": p1, 
            "rec_PER": r1,
            "f1_PER": f1,
            "prec_ORG": p2, 
            "rec_ORG": r2,
            "f1_ORG": f2,
            "prec_LOC": p3, 
            "rec_LOC": r3,
            "f1_LOC": f3,
            "prec_MISC": p4, 
            "rec_MISC": r4,
            "f1_MISC": f4}

