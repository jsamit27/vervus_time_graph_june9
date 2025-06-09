from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


def get_column_mapping_with_llm(column_names):
    prompt = f"""
You are a data understanding assistant.

Given the following list of column names from a real estate dataset:
{column_names}

Please identify which one most likely corresponds to:
1. ClosePrice (the final price at which a property was sold)
2. DaysOnMarket (how long the property stayed listed before being sold)

Return the result in this format:
ClosePrice: <column name>
DaysOnMarket: <column name>
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = response.choices[0].message.content
    lines = text.strip().splitlines()
    result = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result

@app.get("/")
def root():
    return {"message": "FastAPI CSV Plot Service is running. Visit /docs to test the API."}


@app.post("/plot")
async def plot_graph(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        print("📊 Columns in CSV:", df.columns.tolist())

        mapping = get_column_mapping_with_llm(list(df.columns))
        print("🤖 LLM Mapping:", mapping)

        close_col = mapping.get("ClosePrice")
        dom_col = mapping.get("DaysOnMarket")

        if close_col not in df.columns or dom_col not in df.columns:
            raise HTTPException(status_code=400, detail="Mapped columns not found in CSV.")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[dom_col], df[close_col], alpha=0.6)
        ax.set_xlabel(dom_col)
        ax.set_ylabel(close_col)
        ax.set_title("Close Price vs Days on Market")
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        print("✅ Plot generated successfully.")
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
