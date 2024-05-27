TEMPLATE_TO_KEY_TABLE_SYSTEM = "You are a useful assistant."
TEMPLATE_TO_KEY_TABLE_PROMPT = """Extract key information (detailed) according to the following template requirements and integrate it into a table for collecting essential project information from users. Include as many keys as possible with as much detail as possible and refine the table as much as possible:

Template Requirements:
{template_requirements}

respond in JSON format:
{{
    'name of key1 in str format': 'simple description of key1 in str format',
    'name of key2 in str format': 'simple description of key2 in str format',
}}
Note: JSON format cannot contain nested structures; it should only have one layer, with different names representing different keys."""