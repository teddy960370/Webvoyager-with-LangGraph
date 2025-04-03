import json
import re
import datetime
import random
import os
import pandas as pd
from dateutil import parser
from dateutil.relativedelta import relativedelta

class TimeTemplateReplacer:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._read_json()

    def _read_json(self):
        """讀取JSON檔案"""
        with open(self.json_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        data = []
        for line in lines:
            if line.strip():
                data.append(json.loads(line))
        return data

    def get_replaced_dates(self, seed=None, min_days=30, max_days=180):
        """替換問題中的日期模板為未來半年內的隨機日期，支持日期偏移"""
        if seed is not None:
            random.seed(seed)
        
        # 取得當前日期
        today = datetime.datetime.now()
        
        # 複製資料以避免修改原始資料
        replaced_data = []
        
        # 定義正則表達式模式來匹配模板和可能的日期偏移
        template_pattern = r'(YYYY_TEMPLATE|MM_TEMPLATE|DD_TEMPLATE|YYYY_MM_TEMPLATE|MM_DD_TEMPLATE|YYYY_MM_DD_TEMPLATE)([+-]\d+)?'
        
        for item in self.data:
            question = item.get('ques', '')
            
            # 檢查是否有任何模板匹配
            if re.search(template_pattern, question):
                # 生成一個未來隨機天數作為基準日期
                future_days = random.randint(min_days, max_days)
                base_date = today + datetime.timedelta(days=future_days)
                
                # 找到所有模板匹配並替換
                def replace_template(match):
                    template = match.group(1)
                    offset_str = match.group(2) or ""
                    
                    # 解析日期偏移量
                    offset_days = 0
                    if offset_str:
                        offset_days = int(offset_str)
                    
                    # 計算最終日期
                    final_date = base_date + datetime.timedelta(days=offset_days)
                    
                    # 根據模板類型，格式化日期
                    if template == 'YYYY_TEMPLATE':
                        return final_date.strftime('%Y')
                    elif template == 'MM_TEMPLATE':
                        return final_date.strftime('%m')
                    elif template == 'DD_TEMPLATE':
                        return final_date.strftime('%d')
                    elif template == 'YYYY_MM_TEMPLATE':
                        return final_date.strftime('%B, %Y')
                    elif template == 'MM_DD_TEMPLATE':
                        return final_date.strftime('%B %d')
                    elif template == 'YYYY_MM_DD_TEMPLATE':
                        return final_date.strftime('%B %d, %Y')
                    return match.group(0)  # 如果無法識別模板，保持原樣
                
                # 使用正則表達式替換所有匹配項
                question = re.sub(template_pattern, replace_template, question)
            
            # 創建替換後的新項目
            new_item = item.copy()
            new_item['ques'] = question
            replaced_data.append(new_item)
        
        return replaced_data

# 使用範例
def main():
    json_path = r"e:/碩士/論文/Webvoyager-with-LangGraph/data/test2_preprocessed.jsonl"
    replacer = TimeTemplateReplacer(json_path)
    
    # 設定隨機種子以獲得可重現的結果
    replaced_data = replacer.get_replaced_dates(seed=42)
    
    # 顯示原始問題和替換後的問題比較
    print("原始問題 vs 替換後問題 (前5項):")
    for i, (orig, replaced) in enumerate(zip(replacer.data[:5], replaced_data[:5])):
        print(f"\n範例 {i+1}:")
        print(f"原始: {orig['ques']}")
        print(f"替換: {replaced['ques']}")
    
    # 將結果保存為新的JSONL檔案
    output_path = r"e:/碩士/論文/Webvoyager-with-LangGraph/data/test2_preprocessed_V2.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in replaced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n替換後資料已儲存至: {output_path}")

if __name__ == "__main__":
    main()
