import json

def parse_json_with_fix(json_string, retries=3):
    """
    Attempts to parse a JSON string, with retries and basic fixing for common issues.
    """
    for i in range(retries):
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error (attempt {i+1}/{retries}): {e}")
            # Attempt to fix common issues
            if "Unterminated string" in str(e) or "Expecting ',' delimiter" in str(e):
                if json_string.endswith('}'):
                    json_string += ']'
                elif json_string.endswith(']'):
                    json_string += '}'
                else:
                    json_string += '}'
            elif "Expecting property name enclosed in double quotes" in str(e):
                # This is harder to fix automatically without more context
                pass
            
            if i < retries - 1:
                print("Attempting to fix JSON and retry...")
            else:
                raise # Re-raise if all retries fail

    return json.loads(json_string) # Should not be reached if retries fail
