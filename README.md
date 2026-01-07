# pydantic-ai-excercises
Set of excercises to learn Pydantic AI

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Exercise 1

Before running, create a `.env` file based on the template (.env.template) with your OpenRouter API key:

```bash
echo "OPENROUTER_API_KEY=your_openrouter_api_key_here" > .env
```

Then run the customer support agent:

```bash
python excercise/customer_support_agent.py
```
