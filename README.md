
# **Samsa AI: Literature Essay Generator**

This project uses OpenAI's GPT-based models to generate a comparative analysis essay based on the theme of social isolation across multiple books.

## Setup Instructions

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
```

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
Add your OpenAI API key to the environment variables:

```bash
export OPENAI_API_KEY=your_api_key_here
```

### 3. Add books
Add any books you want to analyze to the `data/books` directory.


### 4. Run the Application
Execute the main script to process the books and generate the essay:

```bash
python src/main.py
```

## Notes
### Rate limit
GPT-4-mini has a rate limit of 200K tokens per minute. For perspective, Bell Jar's raw file is 107K tokens. So running this program multiple times in a short period can encounter rate limit errors. Further improvement would be needed to handle this.
