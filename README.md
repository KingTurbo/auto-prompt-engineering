# Prompt Engineering Pipeline

This repository contains a Python script (`prompt_engin.py`) that orchestrates a multi-step prompt engineering pipeline using the Google Gemini API. Its purpose is to take a user input and generate a comprehensive, structured prompt by breaking down the request into several key components: role, task, specification, context, and examples.

## How It Works

The `prompt_engin.py` script automates the process of crafting detailed prompts for large language models (LLMs) by chaining together several specialized "engines," each responsible for a specific aspect of the prompt.

### Core Functionality

The `call_gemini_api` function is the central component for interacting with the Google Gemini API. It handles sending user parts, applying system instructions, and enforcing a predefined response schema to ensure structured JSON output from the model. It also includes error handling for JSON decoding and other unexpected API issues.

### Pipeline Steps

The script executes a sequence of calls to the Gemini API, each guided by a specific system instruction file, to build up the final prompt structure:

1.  **Role Generation** (using `101role_system_instruction.txt`):
    This step defines an expert role for the LLM based on the initial user input. It detects whether the input is related to code (e.g., Python, JavaScript) or general research/explanation, and then crafts a suitable expert persona (e.g., "Python logging expert," "renowned professor of quantum physics").

2.  **Task Generation** (using `102task_system_instruction.txt`):
    Following role definition, this step extracts the core task from the user input and decomposes it into a clear, ordered chain of thought or "thought steps." For coding tasks, it focuses on bug diagnosis and targeted fixes; for research tasks, it outlines information gathering, synthesis, and presentation.

3.  **Specification Generation** (using `103specification_system_instruction.txt`):
    This crucial step generates a "Specifications" section that lists all critical details, placeholders, and formatting rules. It emphasizes the dire importance of adhering to these specifications, often using emotionally charged language to underscore the non-negotiable nature of the requirements.

4.  **Context Generation** (using `104context_system_instruction.txt`):
    This step provides a comprehensive "Context" for the prompt. It includes a high-level overview, explains the business/task context (why the task matters), the system context (how this prompt fits into the broader automated pipeline), the emotional importance (what's at stake), and the final usage of the generated output.

5.  **Example Generation** (using `105example_system_instruction.txt`):
    The final step generates between 1 and 5 paired "user instruction" and "example output" entries. These examples illustrate how a user might phrase a request and what the LLM should return, matching the formality, placeholders, and structure defined in the specifications.

### Input Files

*   `input.txt`: This file is used to provide the primary user input that kicks off the prompt generation process.
*   ``specification.txt``: This file can contain highly relevant information. Something like: Do not provide any comments withing the generated code. The data is used for "Specification Generation".
*   `notes.txt`: This file can contain external notes or additional information that might be relevant to the prompt generation, which will be included in the final output and is not processed to the llm it is just appended to the final crafted prompt.



### Output

The script generates a markdown file named `generated_summary_YYYYMMDD_HHMMSS_ms.md` (e.g., `generated_summary_20250621_211315_123.md`). This file consolidates all the generated sections (role, task, specification, context, examples, notes, and original user input) into a single, well-structured markdown document.

## Configuration

The `prompt_engin.py` script allows for several configurable parameters:

*   **`API_KEY`**: Your Google Gemini API key.
    ```python
    API_KEY = "YOUR_API_KEY_HERE" # Replace "xxx" with your actual API key
    ```
*   **`GEMINI_MODEL`**: The specific Gemini model to be used for API calls.
    ```python
    GEMINI_MODEL = "gemini-2.5-flash" # Can be changed to other available Gemini models
    ```
*   **System Instruction File Paths**: The paths to the individual system instruction files can be configured if they are moved or renamed.
    ```python
    ROLE_SYSTEM_INSTRUCTION_FILE = "./system_instructions/101role_system_instruction.txt"
    TASK_SYSTEM_INSTRUCTION_FILE = "./system_instructions/102task_system_instruction.txt"
    SPECIFICATION_SYSTEM_INSTRUCTION_FILE = "./system_instructions/103specification_system_instruction.txt"
    CONTEXT_SYSTEM_INSTRUCTION_FILE = "./system_instructions/104context_system_instruction.txt"
    EXAMPLE_SYSTEM_INSTRUCTION_FILE = "./system_instructions/105example_system_instruction.txt"
    ```

## Usage

To run the script, ensure you have the necessary Python packages installed (e.g., `google-generativeai`). You can typically install them via pip:

```bash
pip install google-generativeai
```

Then, execute the script from your terminal:

```bash
python prompt_engin.py
```

The script will read input from `input.txt` and `specification.txt`, call the Gemini API in sequence for each prompt engineering step, and finally generate a markdown summary file in the current directory.
