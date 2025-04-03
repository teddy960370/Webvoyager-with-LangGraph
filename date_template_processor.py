import json
import os
import time
from openai import OpenAI
from datetime import datetime
import logging
import argparse
import pandas as pd
import re

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_with_openai(question, client, model):
    """
    使用OpenAI API分析問題中的日期並替換為模板
    """
    prompt = f"""
    Please analyze the date content in the following question:
    
    "{question}"
    """
    
    try:

        SYSTEM_PROMPT = """
You are an assistant for detecting and replacing date formats.

The user will provide a series of text descriptions, and you will replace dates in the text with specific templates while preserving the original meaning.

Strictly follow these rules:

1. Check if the text contains any dates. If not, return the following JSON:
   ```json
   {
     "has_dates": false,
     "reason": "why you replace or not"
   }
   ```

2. Do not replace inferred dates, such as "next 2 months" or "following days."

3. Identify all dates in the text and replace them using templates based on their format:
   - Only year: replace with `YYYY_TEMPLATE`
   - Only month: replace with `MM_TEMPLATE`
   - Only day: replace with `DD_TEMPLATE`
   - Year and month: replace with `YYYY_MM_TEMPLATE`
   - Month and day: replace with `MM_DD_TEMPLATE`
   - Full date: replace with `YYYY_MM_DD_TEMPLATE`
   
   Return the following JSON:
   ```json
   {
     "has_dates": true,
     "all_old_dates": false,
     "modified_question": "task with replaced dates",
     "earliest_date": "identified reference date",
     "reason": "why you replace or not"
   }
   ```

4. If there are multiple dates, find the earliest date, calculate the number of days between the earliest and other dates, and replace the other dates using the template + days from the earliest date.

# Steps

- Analyze the text to extract any date information.
- Categorize each extracted date into one of the defined formats.
- Replace dates with respective templates.
- Calculate differences in days if multiple dates are present, and apply template + days format.

# Output Format

The output should be in JSON format, either indicating no dates were found or providing a modified version of the text with replaced date templates. Include the reason for any replacement decisions.

# Examples

### Example 1

**Task:** Compare prices for economy class round-trip flights from Dubai to Rome, departing on March 1, 2024, and returning on March 8, 2024, and select the option with the fewest stops.

**Result:** 
```json
{
  "has_dates": true,
  "all_old_dates": false,
  "modified_question": "Compare prices for economy class round-trip flights from Dubai to Rome, departing on YYYY_MM_DD_TEMPLATE, and returning on YYYY_MM_DD_TEMPLATE+7, and select the option with the fewest stops.",
  "earliest_date": "March 1, 2024",
  "reason": "Replaced exact dates with templates and calculated day difference."
}
```

### Example 2

**Task:** Search a one-way flight from Dublin To Athens Greece for 1 Adult that leaves on December 30 and analyse the price graph for the next 2 months.

**Result:** 
```json
{
  "has_dates": true,
  "all_old_dates": false,
  "modified_question": "Search a one-way flight from Dublin To Athens Greece for 1 Adult that leaves on MM_DD_TEMPLATE and analyse the price graph for the next 2 months.",
  "earliest_date": "December 30",
  "reason": "Replaced specific date with a template while ignoring inferred future timeframe."
}
```
"""
        response = client.chat.completions.create(
            model=model,  # 使用參數傳入的模型
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content.strip()
        
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error(f"無法解析API回應為JSON: {result}")
            return {"has_dates": False}
            
    except Exception as e:
        logger.error(f"調用OpenAI API時發生錯誤: {e}")
        return {"has_dates": False}

def process_file(input_path, output_path, client, model):
    """
    處理JSONL文件，替換日期為模板
    """
    logger.info(f"開始處理檔案: {input_path}")
    
    # 讀取JSONL檔案
    with open(input_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]
    
    modified_count = 0
    processed_data = []
    
    for i, item in enumerate(data):
        logger.info(f"處理項目 {i+1}/{len(data)}: {item['id']}")
        question = item["ques"]
        
        # 使用OpenAI分析問題中的日期
        result = process_with_openai(question, client, model)
        
        # 根據分析結果更新問題
        if result.get("has_dates", False) and not result.get("all_old_dates", True):
            item["ques"] = result.get("modified_question", question)
            logger.info(f"已更新問題: {item['ques']}")
            modified_count += 1
        else:
            logger.info("無需更改")
        
        processed_data.append(item)
    
    # 將處理後的資料寫入新檔案
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in processed_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"處理完成。共修改 {modified_count}/{len(data)} 項目。")
    logger.info(f"輸出已儲存至: {output_path}")

def compare_and_export_excel(input_path, output_path, excel_output_path):
    """
    分析輸入和輸出文件的差異，並匯出成Excel
    
    Args:
        input_path (str): 原始JSONL文件路徑
        output_path (str): 處理後JSONL文件路徑
        excel_output_path (str): 匯出Excel的路徑
    """
    logger.info(f"開始比較文件差異: {input_path} 與 {output_path}")
    
    # 讀取輸入和輸出文件
    input_data = {}
    output_data = {}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                input_data[item['id']] = item['ques']
                
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                output_data[item['id']] = item['ques']
    
    # 定義用來檢測模板的正則表達式
    template_patterns = {
        "YYYY_TEMPLATE": r'YYYY_TEMPLATE(?:\+\d+)?',
        "MM_TEMPLATE": r'MM_TEMPLATE(?:\+\d+)?',
        "DD_TEMPLATE": r'DD_TEMPLATE(?:\+\d+)?',
        "YYYY_MM_TEMPLATE": r'YYYY_MM_TEMPLATE(?:\+\d+)?',
        "MM_DD_TEMPLATE": r'MM_DD_TEMPLATE(?:\+\d+)?',
        "YYYY_MM_DD_TEMPLATE": r'YYYY_MM_DD_TEMPLATE(?:\+\d+)?'
    }
    
    # 準備Excel數據
    excel_data = []
    
    for item_id in input_data:
        original_ques = input_data[item_id]
        modified_ques = output_data.get(item_id, original_ques)
        
        # 檢查是否有修改
        is_modified = original_ques != modified_ques
        
        # 檢測使用的模板
        template_usage = {}
        for template_name, pattern in template_patterns.items():
            matches = re.findall(pattern, modified_ques)
            template_usage[template_name] = matches
        
        # 填充資料列
        row = {
            "是否有修改": "是" if is_modified else "否",
            "原始任務": original_ques,
            "修改後任務": modified_ques
        }
        
        # 填充模板使用情況
        for i, (template_name, matches) in enumerate(template_usage.items(), 1):
            row[f"使用模板{i}"] = ", ".join(matches) if matches else ""
        
        excel_data.append(row)
    
    # 創建DataFrame並匯出成Excel
    df = pd.DataFrame(excel_data)
    df.to_excel(excel_output_path, index=False)
    
    logger.info(f"比較完成，結果已匯出至: {excel_output_path}")
    return len([row for row in excel_data if row["是否有修改"] == "是"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dates in questions and replace them with templates")
    parser.add_argument("--api_key", required=True, type=str, help="OpenAI API key")
    parser.add_argument("--api_model", default="gpt-4o", type=str, help="OpenAI model name to use")
    parser.add_argument("--input_file", required=True, type=str, help="Path to input JSONL file")
    parser.add_argument("--output_file", required=True, type=str, help="Path to output JSONL file")
    parser.add_argument("--analyze", action="store_true", help="Analyze differences and export to Excel",default=True)
    parser.add_argument("--excel_output", type=str, help="Path to export Excel analysis (required if --analyze is used)")
    
    args = parser.parse_args()
    
    # 初始化OpenAI客戶端
    client = OpenAI(api_key=args.api_key)
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 處理文件
    process_file(args.input_file, args.output_file, client, args.api_model)
    
    # 如果需要分析差異，匯出Excel
    if args.analyze:
        if not args.excel_output:
            parser.error("--excel_output is required when using --analyze")
        
        os.makedirs(os.path.dirname(args.excel_output), exist_ok=True)
        modified_count = compare_and_export_excel(args.input_file, args.output_file, args.excel_output)
        logger.info(f"分析結果：共有 {modified_count} 個問題被修改")
