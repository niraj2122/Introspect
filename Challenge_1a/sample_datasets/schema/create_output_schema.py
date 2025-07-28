import json

output_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "outline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "string"},
                    "text": {"type": "string"},
                    "page": {"type": "integer"}
                },
                "required": ["level", "text", "page"]
            }
        }
    },
    "required": ["title", "outline"]
}

with open("output_schema.json", "w") as f:
    json.dump(output_schema, f, indent=2)

print("output_schema.json has been created.")
