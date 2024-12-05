# Docx Translator with OpenAI

This repository provides a Python script for translating `.docx` files using OpenAI's GPT(-4) API.

---

## Features

- Supports translation between any pair of languages (configurable in the script).
- Handles large texts by splitting them into manageable chunks that fit OpenAI's token limits.
- Maintains contextual continuity by incorporating previous translations and source text into the translation process.
- Allows the use of a sample translation file to guide translation tone and style. (The current script focuses on formal equivalence, aiming to preserve the original meaning, style, and structure of the text.)
- Reads input from `.docx` files and writes translated output to `.docx` files while preserving paragraph structure.
- Includes error handling and retry mechanisms for API calls.

---

## Requirements

1. Python 3.7 or higher.
2. Required libraries:
   - `openai`
   - `python-docx`
   - `tiktoken`
3. A valid OpenAI API key.

To install the required libraries, run:

```bash
pip install openai python-docx tiktoken
## Setup

### Configuration File

Create a `config.json` file in the same directory as the script and add your OpenAI API key:

```json
{
    "OPENAI_API_KEY": "your-api-key-here"
}
```

### Input and Output Files

- Prepare the `.docx` file you want to translate.
- Optionally, create a `.docx` file with sample translations to guide the translation style.

---

## Usage

1. Open the script and modify the `main()` function call:

   ```python
   main('input.docx', 'output.docx', sample_translation_file='sample_translation.docx')
   ```

   - Replace `input.docx` with the path to your input file.
   - Replace `output.docx` with the desired output file path.
   - Optionally, specify a sample translation file.

2. Run the script:

   ```bash
   python openai_translator.py
   ```

3. The translated document will be saved as the specified output file.



