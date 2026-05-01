import ollama
import json

PROMPT = """
You are extracting structured oncology timeline data from clinical notes.

Return JSON only with this exact schema:
{
  "events": [
    {
      "condition_type": "treatment" or "irAE",
      "condition": "short clinical label",
      "start_date": "YYYY-MM-DD or YYYY-MM or YYYY",
      "end_date": "YYYY-MM-DD or YYYY-MM or YYYY or null"
    }
  ]
}

Rules:
1. Include all dated immunotherapy-related treatments and irAEs.
2. `treatment` includes:
- checkpoint inhibitors (e.g., pembrolizumab, nivolumab, atezolizumab)
- other cancer therapies (e.g., chemotherapy, targeted therapy)
- irAE-directed management (e.g., steroids, IV methylprednisolone, prophylaxis)
3. `irAE` includes immune-related toxicities/events only.
4. Sometimes there may be multiple entries per treatment or condition if there are multiple distinct time periods (e.g., treatment stopped and restarted, irAE resolved and recurred).
5. If a date range is implied, set `start_date` and `end_date`.
6. If only onset is known, set `start_date` and `end_date` to null.
7. Preserve date precision from source text (day > month > year).
8. Use concise condition names (e.g., "Atezolizumab", "Hepatitis").
9. Do not add keys beyond the schema.
10. ONLY return JSON, no explanations or extra text.
"""

def extract_events(model, temperature, note):
    """Function to extract structured events from a clinical note using the LLM.
    Returns a list of event dictionaries with keys: condition_type, condition, start_date, end_date."""
    reponse = ollama.chat(
        model = model, 
        format = 'json',
        options = {'temperature': temperature},
        messages = [
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': note},
        ]
    )
    data = json.loads(reponse['message']['content'])
    events = data.get('events', []) if isinstance(data, dict) else []
    out = []
    for e in events:
        ctype = e.get('condition_type', '').strip().lower()
        condition = e.get('condition', '').strip().lower()
        start = e.get('start_date', '')
        end = e.get('end_date', None)

        start_clean = str(start).strip()
        end_clean = None if end is None else str(end).strip()
        if end_clean and end_clean.lower() in {"none", "null", "na", "n/a", ""}:
            end_clean = None

        out.append(
            {
                'condition_type': ctype,
                'condition': condition,
                'start_date': start_clean,
                'end_date': end_clean,
            }
        )

    return out
