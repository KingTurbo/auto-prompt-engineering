import json
from google import genai
from google.genai import types
from typing import Dict, Any, List
import time
import datetime

API_KEY = "xxx"

GEMINI_MODEL = "gemini-2.5-flash" 

ROLE_SYSTEM_INSTRUCTION_FILE = "./system_instructions/101role_system_instruction.txt"
TASK_SYSTEM_INSTRUCTION_FILE = "./system_instructions/102task_system_instruction.txt"
SPECIFICATION_SYSTEM_INSTRUCTION_FILE = "./system_instructions/103specification_system_instruction.txt"
CONTEXT_SYSTEM_INSTRUCTION_FILE = "./system_instructions/104context_system_instruction.txt"
EXAMPLE_SYSTEM_INSTRUCTION_FILE = "./system_instructions/105example_system_instruction.txt"

def _load_system_instruction(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_list_from_file(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        items = [line.strip() for line in f.readlines()]
    return items


def call_gemini_api(
    user_parts: List[types.Part],
    system_instruction_path: str,
    response_schema: types.Schema,
    api_key: str,
    model_name: str = GEMINI_MODEL
) -> Dict[str, Any]:

    client = genai.Client(api_key=api_key)

    system_instruction_text = _load_system_instruction(system_instruction_path)

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=[types.Part.from_text(text=system_instruction_text)],
    )

    full_response_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=[types.Content(role="user", parts=user_parts)],
            config=generate_content_config,
        ):
            full_response_text += chunk.text
        return json.loads(full_response_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from model response: {e}")
        return {} 
    except Exception as e:
        print(f"An unexpected error occurred during content generation: {e}")
        return {}

def generate_role(user_input: str) -> Dict[str, Any]:
    print("--- Calling Gemini for Role Generation ---")
    
    role_response_schema = types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "role": genai.types.Schema(type=genai.types.Type.STRING),
            "code": genai.types.Schema(type=genai.types.Type.BOOLEAN),
            "research": genai.types.Schema(type=genai.types.Type.BOOLEAN)
        },
    )

    user_parts = [types.Part.from_text(text=user_input)]
    return call_gemini_api(
        user_parts=user_parts,
        system_instruction_path=ROLE_SYSTEM_INSTRUCTION_FILE,
        response_schema=role_response_schema,
        api_key=API_KEY
    )

def generate_task(user_input: str, role_str: str) -> Dict[str, Any]:

    print("--- Calling Gemini for Task Generation ---")

    task_response_schema = types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "task": genai.types.Schema(type=genai.types.Type.STRING),
        },
    )

    combined_input_for_task_model = f"Role: \"{role_str}\"\nUser Input: \"{user_input}\""
    user_parts = [types.Part.from_text(text=combined_input_for_task_model)]

    return call_gemini_api(
        user_parts=user_parts,
        system_instruction_path=TASK_SYSTEM_INSTRUCTION_FILE,
        response_schema=task_response_schema,
        api_key=API_KEY
    )

def generate_specification(user_input: str, role_str: str, task: str, specific_formatted: list) -> Dict[str, Any]:

    print("--- Calling Gemini for Specifiaction Generation ---")

    response_schema= types.Schema(
        type = genai.types.Type.OBJECT,
        properties = {
            "specification": genai.types.Schema(type = genai.types.Type.STRING),
        },
    )


    combined_input_for_specification_model = (
        f"Role: \"{role_str}\"\n"
        f"User Input: \"{user_input}\"\n"
        f"Task: \"{task}\"\n"
        f"Specific Instructions:\n{specific_formatted}"
    )

    user_parts = [types.Part.from_text(text=combined_input_for_specification_model)]

    return call_gemini_api(
        user_parts=user_parts,
        system_instruction_path=SPECIFICATION_SYSTEM_INSTRUCTION_FILE,
        response_schema=response_schema,
        api_key=API_KEY
    )


def generate_context(user_input: str, role_str: str, task: str, specification: str) -> Dict[str, Any]:

    print("--- Calling Gemini for Context Generation ---")

    response_schema=types.Schema(
        type = genai.types.Type.OBJECT,
        properties = {
            "overview": genai.types.Schema(
                type = genai.types.Type.STRING,
            ),
            "business_context": genai.types.Schema(
                type = genai.types.Type.STRING,
            ),
            "system_context": genai.types.Schema(
                type = genai.types.Type.STRING,
            ),
            "emotional_importance": genai.types.Schema(
                type = genai.types.Type.STRING,
            ),
            "final_usage": genai.types.Schema(
                type = genai.types.Type.STRING,
            ),
        },
    )


    combined_input_for_context_model = (
        f"Role: \"{role_str}\"\n"
        f"User Input: \"{user_input}\"\n"
        f"Task: \"{task}\"\n"
        f"Specification: \"{specification}\"\n"
    )

    user_parts = [types.Part.from_text(text=combined_input_for_context_model)]

    return call_gemini_api(
        user_parts=user_parts,
        system_instruction_path=CONTEXT_SYSTEM_INSTRUCTION_FILE,
        response_schema=response_schema,
        api_key=API_KEY
    )



def generate_examples(user_input: str, role_str: str, task: str, specification: str, context: str) -> Dict[str, Any]:

    print("--- Calling Gemini for Example Generation ---")

    response_schema= types.Schema(
        type = genai.types.Type.OBJECT,
        properties = {
            "examples": genai.types.Schema(
                type = genai.types.Type.STRING,
            )
        }
    )


    combined_input_for_example_model = (
        f"Role: \"{role_str}\"\n"
        f"User Input: \"{user_input}\"\n"
        f"Task: \"{task}\"\n"
        f"Specification: \"{specification}\"\n"
        f"Context: \"{context}\"\n"
    )

    user_parts = [types.Part.from_text(text=combined_input_for_example_model)]

    return call_gemini_api(
        user_parts=user_parts,
        system_instruction_path=EXAMPLE_SYSTEM_INSTRUCTION_FILE,
        response_schema=response_schema,
        api_key=API_KEY
    )



def process_full_request(user_input: str, external_notes="") -> Dict[str, Any]:

    final_output = {
        "user_input": user_input,
        "role": "N/A",
        "is_code": False,
        "is_research": False,
        "task": "N/A",
        "specification": "N/A",
        "context": "N/A",
        "examples": "N/A",
        "notes": []
    }

    role_result = generate_role(user_input)

    if role_result and "role" in role_result:
        final_output["role"] = role_result["role"]
        final_output["is_code"] = role_result.get("code", False)
        final_output["is_research"] = role_result.get("research", False)
        
    else:
        print("Failed to generate a valid role. Aborting task generation.")
        return final_output

    time.sleep(3)

    if final_output["role"] != "N/A": 
        task_result = generate_task(user_input, final_output["role"])
        if task_result and "task" in task_result:
            final_output["task"] = task_result["task"]
            
        else:
            print("Failed to generate a valid task.")

    time.sleep(3)
    if final_output["role"] != "N/A" and final_output["task"] != "N/A":
        example_specific_list = load_list_from_file("!!specification.txt")
        
        spec_result = generate_specification(
            user_input=user_input,
            role_str=final_output["role"],
            task=final_output["task"],
            specific_formatted=example_specific_list  
        )
        if spec_result:
            final_output["specification"] = spec_result.get("specification", "N/A")
            
        else:
            final_output["specification"] = "N/A"
            print("Failed to generate valid specification.")


    time.sleep(3)

    if final_output["role"] != "N/A" and final_output["task"] != "N/A" and final_output["specification"] != "N/A" :
        context_result = generate_context(
            user_input=user_input,
            role_str=final_output["role"],
            task=final_output["task"],
            specification=final_output["specification"]
        )

        if context_result:
            final_output["context"] = context_result
            
        else:
            final_output["context"] = "N/A"
            print("Failed to generate valid context.")

    time.sleep(3)

    if final_output["role"] != "N/A" and final_output["task"] != "N/A" and final_output["specification"] != "N/A" and final_output["context"] != "N/A":
        if final_output["is_code"]:
            final_output["example"] = user_input
            final_output["user_input"] = ""

        elif final_output["is_research"]:
            final_output["example"] = ""
        else:
            example_results = generate_examples(
                user_input=user_input,
                role_str=final_output["role"],
                task=final_output["task"],
                specification=final_output["specification"],
                context=final_output["context"]
            ) 

            if context_result:
                final_output["examples"] = example_results.get("examples", "N/A")
                
            else:
                final_output["examples"] = "N/A"
                print("Failed to generate valid examples.")
    

    final_output["notes"] = external_notes



    return final_output

if __name__ == "__main__":

    user_input_1 = _load_system_instruction("!!input.txt")

    external_notes_from_file = _load_system_instruction("!!notes.txt")


    print(f"\n--- Processing Input 1: {user_input_1[:50]}... ---")

    output_1 = process_full_request(user_input_1, external_notes=external_notes_from_file)

    final_output = output_1

    print("\n--- Consolidated Output 1 ---")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"

    section_order = [
        "role",
        "task",
        "specification",
        "context",
        "examples",
        "notes",
        "user_input"
    ]

    combined_markdown_content = []

    print(f"\n--- Building combined Markdown file... ---")

    for key_name in section_order:
        value = final_output.get(key_name, "N/A")

        markdown_title = key_name.replace("_", " ").title()

        data_content = ""
        if key_name == "specification" and isinstance(value, dict):
            data_content = f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```"
        

        elif key_name == "context" and isinstance(value, dict):
            sub_sections = []
            for sub_key, sub_val in value.items():
                sub_title = sub_key.replace("_", " ").title()
                sub_sections.append(f"## {sub_title}\n{sub_val}")
            data_content = "\n\n".join(sub_sections)

        elif key_name == "notes" and isinstance(value, list):
            if value:
                data_content = "\n".join([f"- {item}" for item in value])
            else:
                data_content = "N/A"
        elif isinstance(value, (bool, int, float)):
            data_content = str(value)
        elif value is None or (isinstance(value, str) and value.strip() == ""):
            data_content = "N/A"
        else:
            data_content = str(value)

        content_prefix = "\n" if data_content and key_name != "context" else ""
        combined_markdown_content.append(f"# {markdown_title}\n{content_prefix}{data_content}\n====\n")


    final_output_markdown = "\n".join(combined_markdown_content)

    combined_filename = f"generated_summary_{timestamp_str}.md"
    file_path = combined_filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_output_markdown)
    print(f"  Created: {file_path}")

    print(f"\n--- Combined Markdown file generation complete. File saved as '{file_path}' ---")
