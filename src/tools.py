import ollama
### Discover local models for UI
def discover_local_ollama_models():
    """
    Return a sorted list of model identifiers from ollama.list(),
    e.g. ['gemma3:4b', 'gemma3:1b', 'granite4:latest', ...]
    """
    try:
        models = ollama.list()["models"]  # returns a list of Model(...) objects
    except Exception as e:
        # Ollama client not available / Ollama not running
        # Return empty list so UI can show an error and avoid crash
        return []

    names = []
    for m in models:
        # each m has attribute `model` per your printed output
        try:
            names.append(m.model)
        except Exception:
            # defensive: if structure differs, try string conversion and parse
            s = str(m)
            # a safe fallback: try to extract token before first space or comma
            token = s.split()[0].strip(",")
            if token:
                names.append(token)

    # de-duplicate and sort for stable UI behaviour
    return sorted(dict.fromkeys(names))