# Output Schema

## Directory Structure
`data/outputs/{doc_id}/`

## manifest.json
Root metadata file.
```json
{
  "meta": { ... },
  "capabilities": { "ocr": true, "camelot": false },
  "items": [
    {
      "item_id": "fig_0001",
      "type": "figure",
      "ai_annotations": { ... },
      "artifacts": { "preview_png": "..." }
    }
  ]
}
```

## evidence.json
Traceability for every item.
```json
{
  "page_index": 0,
  "bbox": [100, 100, 500, 400],
  "caption_text": "Figure 1: XRD pattern...",
  "snippet_text": "As shown in Figure 1, the peaks correspond to...",
  "extraction_trace": [...]
}
```

## ai.json
AI-enriched metadata.
```json
{
  "subtype": "spectrum",
  "subtype_confidence": 0.8,
  "conditions": [
    { "name": "temperature", "value": "300 K", "source": "regex_heuristic" }
  ],
  "material_candidates": []
}
```
