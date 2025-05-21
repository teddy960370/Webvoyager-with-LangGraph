"""
本地 RAG (檢索增強生成) 功能模組
用於管理知識庫文檔的載入、向量化、相似度計算和檢索
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# 導入 LangChain 的分塊工具
from langchain.text_splitter import MarkdownTextSplitter, MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter

def load_knowledge_documents(directory_path: str) -> List[Dict[str, Any]]:
    """從指定目錄載入知識文檔，支援 Markdown、JSON、JSONL 和 TXT 格式"""
    documents = []
    
    # 檢查目錄是否存在
    if not os.path.exists(directory_path):
        logging.warning(f"知識庫目錄不存在: {directory_path}")
        return documents
    
    # 遍歷目錄中的所有檔案
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # 處理 Markdown 文件
        if filename.endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 為 Markdown 文件建立文檔結構
                    doc = {
                        "content": content,
                        "source": filename,
                        "type": "markdown"
                    }
                    
                    # 嘗試提取標題作為文檔標題
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith('# '):
                            doc["title"] = line.replace('# ', '')
                            break
                    
                    documents.append(doc)
            except Exception as e:
                logging.error(f"載入 Markdown 文件時發生錯誤 {filename}: {str(e)}")
        
        # 處理 JSONL 文件
        elif filename.endswith('.jsonl'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            doc = json.loads(line.strip())
                            doc["type"] = "jsonl"
                            doc["source"] = filename
                            documents.append(doc)
                        except json.JSONDecodeError:
                            logging.warning(f"無法解析 JSONL 行: {line} in {filename}")
            except Exception as e:
                logging.error(f"載入 JSONL 文件時發生錯誤 {filename}: {str(e)}")
        
        # 處理 JSON 文件
        elif filename.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    if isinstance(doc, list):
                        for item in doc:
                            item["type"] = "json"
                            item["source"] = filename
                        documents.extend(doc)
                    else:
                        doc["type"] = "json"
                        doc["source"] = filename
                        documents.append(doc)
            except Exception as e:
                logging.error(f"載入 JSON 文件時發生錯誤 {filename}: {str(e)}")
        
        # 處理 TXT 文件
        elif filename.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        "content": content, 
                        "source": filename,
                        "type": "text"
                    })
            except Exception as e:
                logging.error(f"載入 TXT 文件時發生錯誤 {filename}: {str(e)}")
    
    logging.info(f"從 {directory_path} 載入了 {len(documents)} 個文檔")
    return documents

def get_embeddings(texts: List[str], llm) -> List[List[float]]:
    """使用 LLM 獲取文本的嵌入向量"""
    embeddings = []
    try:
        # 檢測是否為 Google Gemini LLM 模型
        if hasattr(llm, 'client') and 'google' in str(type(llm.client)).lower():
            # 使用 Google 的文本嵌入 API
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            # 獲取 API 金鑰 (從 llm 中提取)
            api_key = None
            if hasattr(llm, 'genai_api_key'):
                api_key = llm.genai_api_key
            elif hasattr(genai, '_configured_api_key'):
                api_key = genai._configured_api_key
                
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-exp-03-07",
                google_api_key=api_key
            )
            
            # 批次處理以提高效率
            embeddings = embeddings_model.embed_documents(texts)
        else:
            # 回退到 OpenAI 的文本嵌入 API
            from langchain_openai import OpenAIEmbeddings
            
            # 使用與主要 LLM 相同的 API 金鑰
            embeddings_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=llm.openai_api_key if hasattr(llm, 'openai_api_key') else None
            )
            
            # 批次處理以提高效率
            embeddings = embeddings_model.embed_documents(texts)
    except Exception as e:
        logging.error(f"獲取嵌入向量時出錯: {str(e)}")
        # 如果出錯，返回隨機嵌入作為回退
        embeddings = [np.random.rand(768).tolist() for _ in texts]  # Gemini 通常使用 768 維向量
    
    return embeddings

def semantic_search(query: str, documents: List[Dict[str, Any]], 
                   embeddings: List[List[float]], llm,
                   top_k: int = 3) -> List[Dict[str, Any]]:
    """使用語義搜索查找相關文檔"""
    if not documents or not embeddings:
        return []
    
    try:
        # 獲取查詢的嵌入向量
        query_embedding = get_embeddings([query], llm)[0]
        
        # 使用 FAISS 進行高效向量搜索
        import faiss
        
        # 確保嵌入是 numpy 數組並轉換為 float32
        embeddings_array = np.array(embeddings, dtype=np.float32)
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # 創建 FAISS 索引
        dimension = len(query_embedding)
        index = faiss.IndexFlatIP(dimension)  # 內積索引，相當於正則化向量的餘弦相似度
        
        # 正則化向量以使用內積進行餘弦相似度搜索
        faiss.normalize_L2(embeddings_array)
        faiss.normalize_L2(query_array)
        
        # 添加向量到索引
        index.add(embeddings_array)
        
        # 搜索
        # 確保 top_k 不超過可用文檔數量
        actual_top_k = min(top_k, len(documents))
        distances, indices = index.search(query_array, actual_top_k)
        
        # 返回前 K 個相關文檔
        return [documents[i] for i in indices[0]]
    except Exception as e:
        logging.error(f"語義搜索時出錯: {str(e)}")
        return []

def extract_document_content(doc: Dict[str, Any]) -> str:
    """從文檔字典中提取內容文本，優先處理 Markdown 內容"""
    content = ""
    
    # 如果是 Markdown 類型，進行專門處理
    if doc.get("type") == "markdown" and "content" in doc:
        markdown_content = doc["content"]
        # 簡單的 Markdown 處理：移除標記符號但保留內容結構
        # 移除代碼塊標記
        import re
        # 移除代碼塊
        markdown_content = re.sub(r'```[\s\S]*?```', '', markdown_content)
        # 移除內聯代碼
        markdown_content = re.sub(r'`([^`]+)`', r'\1', markdown_content)
        # 移除標題符號但保留文本
        markdown_content = re.sub(r'^#+\s+(.+)$', r'\1', markdown_content, flags=re.MULTILINE)
        # 移除粗體和斜體
        markdown_content = re.sub(r'\*\*(.+?)\*\*', r'\1', markdown_content)
        markdown_content = re.sub(r'\*(.+?)\*', r'\1', markdown_content)
        # 移除列表符號
        markdown_content = re.sub(r'^\s*[-*+]\s+', '', markdown_content, flags=re.MULTILINE)
        return markdown_content.strip()
    
    elif doc.get("type") in ["json", "jsonl"]:
        content = json.dumps(doc, ensure_ascii=False)
    else:
        # 合併所有可能包含內容的欄位
        for key in ["content", "instruction", "input", "output", "ques", "ans", "text"]:
            if key in doc and isinstance(doc[key], str):
                content += doc[key] + " "
    
    return content.strip()

def format_document_context(doc: Dict[str, Any]) -> str:
    """將文檔格式化為上下文字符串，增強對 Markdown 內容的處理"""
    context = ""
    
    # 如果有標題，先添加標題
    if "title" in doc:
        context += f"## {doc['title']}\n\n"
    
    # 根據文檔類型進行格式化
    if doc.get("type") == "markdown" and "content" in doc:
        # 為 Markdown 文檔提供更好的格式
        context += f"{doc['content']}\n\n"
    elif doc.get("type") in ["json", "jsonl"]:
        # 處理 JSON 格式，特別是網頁操作記錄
        if "task_description" in doc and "actions" in doc:
            # 添加任務描述
            context += f"### Web Task: {doc['task_description']}\n"

            # 添加序列範圍信息（如果存在）
            if "sequence_range" in doc:
                context += f"**Actions Sequence**: {doc['sequence_range']}\n"
            
            # 添加操作序列
            context += "\n**Actions**:\n"
            
            for action in doc.get("actions", []):
                action_type = action.get("type", "")
                order = action.get("order", "")
                
                if action_type == "Navigate":
                    url = action.get("url", "")
                    title = action.get("page_title", "")
                    context += f"- [{order}] Navigate to {url} ({title})\n"
                    
                elif action_type == "Click":
                    element_info = action.get("element_info", {})
                    element_text = action.get("elements_text", "")
                    tag = element_info.get("tagName", "")
                    text = element_info.get("text", "")
                    context += f"- [{order}] Click {tag}: {text or element_text}\n"
                    
                elif action_type == "Type":
                    input_value = action.get("input_value", "")
                    element_text = action.get("elements_text", "")
                    context += f"- [{order}] Type: {input_value or element_text}\n"
                    
                elif action_type == "answer":
                    element_text = action.get("elements_text", "")
                    context += f"- [{order}] Complete: {element_text}\n"
                    
                else:
                    context += f"- [{order}] {action_type}\n"
        else:
            # 對於其他JSON格式，嘗試優雅地序列化
            try:
                # 移除type和source等RAG系統添加的字段
                display_doc = {k: v for k, v in doc.items() if k not in ["type", "source", "chunk_id", "total_chunks"]}
                import json
                context += json.dumps(display_doc, ensure_ascii=False, indent=2)
                context += "\n"
            except:
                # 如果序列化失敗，嘗試直接使用字符串表示
                context += str(doc) + "\n"
    elif "ques" in doc and "ans" in doc:
        context += f"Question: {doc['ques']}\nAnswer: {doc['ans']}\n\n"
    elif "instruction" in doc and "output" in doc:
        context += f"Instruction: {doc['instruction']}\nOutput: {doc['output']}\n\n"
    elif "content" in doc:
        context += f"{doc['content']}\n\n"
    elif "text" in doc:
        context += f"{doc['text']}\n\n"
    
    # 添加來源信息（如果存在）
    if "source" in doc:
        context += f"Source: {doc['source']}\n\n"
        
    return context

def generate_optimized_query(task: str, domain: str, llm) -> str:
    """
    使用 LLM 生成更優化的查詢語句，專注於 Web 操作的需求
    """
    if llm is None:
        return f"How to complete the task of '{task}' on '{domain}' website"
    
    try:
        from langchain.prompts import ChatPromptTemplate
        
        # 優化查詢的提示模板
        template = """你是一個專業的網頁導航助手。
        我需要在以下網站完成特定任務，請幫我生成一個精確的查詢語句，用於從操作手冊檢索相關指南：
        
        任務: {task}
        網站: {domain}
        
        請依據你對此網頁的常識，分析任務意圖
        思考並生成完成該任務可能需要哪些功能，或是可能會遭遇的問題
        最後生成相關索引關鍵字協助用於檢索操作手冊中的內容。
        請注意，通常手冊不會包含任務中的指定關鍵字。
        例如，對於“在 Google 上搜索neural networks for image processing”，手冊中可能不會包含“neural networks for image processing”這個詞。
        
        """

        template = """
        Assume the role of a retrieval assistant. When a user provides a web-related task, analyze the task's intent and decide how to retrieve information from a corresponding operations manual. Remember, the manual typically won't include keywords specified in the task.

Analyze the user-given task and map it to potential retrieval actions using the general principles outlined in the manual.

# Steps
1. **Understand User Intent:** Identify the core task the user wants to accomplish.
2. **Generalize Keywords:** Convert specific keywords in the task into general concepts that can be found in the manual.
3. **Search Manual:** Identify relevant sections or instructions in the manual that align with these general concepts.
4. **Formulate Retrieval Queries:** Craft precise queries or actions needed based on the manual's contents to accomplish the user's task.

# Output Format
Provide clear instructions on which sections or topics of the manual the retrieval actions correspond to. Present the output in a concise list format, corresponding to each relevant action or inquiry.

# Examples

- **Input Example:**
  - **Task:** In Google, search for news related to neural networks for image processing within the year 2024 and compile the top 5 articles.
  - **Website:** www.google.com

- **Output Example:**
  - Methods for Google searches.
  - Procedures for querying results within specified timeframes.

# Notes
- Consider edge cases where the manual may not have direct instructions but use closely related sections to infer guidance.
- Ensure the retrieval actions are actionable and directly align with the general principles of the inquiries.
        """

        user_prompt = f"""I need to complete the task of '{task}' on the '{domain}' website. What are the most likely methods to achieve this? \n"""

                
        # 使用 OpenAI API 生成回應
        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": user_prompt}
        ]

        answer = llm.invoke(messages)
        
        optimized_query = answer.content.strip()
        logging.info(f"生成優化查詢: {optimized_query}")
        return optimized_query
        
    except Exception as e:
        logging.error(f"生成優化查詢時出錯: {str(e)}")
        return f"How to complete the task of '{task}' on '{domain}' website"

def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根據文檔類型進行結構化分塊
    特別對 Markdown 文件進行基於標題和段落的結構化分塊
    對 JSON 文件進行基於結構的分塊
    """
    chunked_docs = []
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
    ]

    # 為 Markdown 設置特定的分塊規則，確保標題結構被保留
    markdown_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # 不移除標題以保持文檔結構
    )
    
    # 一般文本的備用分塊器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    
    for doc in documents:
        # 獲取文檔類型和內容
        doc_type = doc.get("type", "")
        content = ""
        
        # 根據文檔類型提取內容
        if doc_type == "markdown" and "content" in doc:
            content = doc["content"]
            # 先按標題進行切分，再對每個部分進行大小限制切分
            header_chunks = markdown_header_splitter.split_text(content)
            chunks = []
            for header_chunk in header_chunks:
                # 對每個標題區塊再次用標準分塊器處理
                chunk_text = header_chunk.page_content if hasattr(header_chunk, 'page_content') else str(header_chunk)
                sub_chunks = text_splitter.split_text(chunk_text)
                chunks.extend(sub_chunks)
        elif doc_type in ["json", "jsonl"] and isinstance(doc, dict):
            # 處理 JSON/JSONL 格式文檔
            # 策略1: 按照 JSON 對象的主要字段進行分塊
            processed_doc = preprocess_web_action_json(doc)
            chunks = chunk_json_document(processed_doc, text_splitter)
        elif "text" in doc:
            content = doc["text"]
            chunks = text_splitter.split_text(content)
        else:
            # 如果沒有識別到明確的內容欄位，嘗試提取所有文本內容
            content = extract_document_content(doc)
            if content:
                chunks = text_splitter.split_text(content)
            else:
                # 如果無法提取內容，則跳過此文檔
                continue
        
        # 為每個分塊創建新的文檔條目
        for i, chunk in enumerate(chunks):
            # 創建新的文檔字典，保留原文檔的元數據
            chunked_doc = doc.copy()
            
            # 更新內容為當前分塊
            if doc_type == "markdown":
                chunked_doc["content"] = chunk
            elif doc_type in ["json", "jsonl"]:
                # 對於 JSON 分塊，保持其原始結構
                if isinstance(chunk, dict):
                    for key, value in chunk.items():
                        chunked_doc[key] = value
                else:
                    chunked_doc["content"] = chunk
            elif "content" in doc:
                chunked_doc["content"] = chunk
            elif "text" in doc:
                chunked_doc["text"] = chunk
            
            # 添加分塊信息
            chunked_doc["chunk_id"] = i
            chunked_doc["total_chunks"] = len(chunks)
            
            # 設置分塊標題（如果是 Markdown）
            if doc_type == "markdown":
                # 檢查分塊是否包含標題（可能是 MarkdownHeaderTextSplitter 已提取的元數據）
                if isinstance(chunk, dict) and "header" in chunk:
                    # 使用 MarkdownHeaderTextSplitter 提供的標題元數據
                    chunked_doc["chunk_title"] = chunk["header"].strip()
                else:
                    # 如果標題不在元數據中，嘗試從文本內容中提取
                    import re
                    headline_match = re.search(r'^#+\s+(.+)$', chunk, re.MULTILINE)
                    if headline_match:
                        chunked_doc["chunk_title"] = headline_match.group(1).strip()
                    elif "title" in doc:
                        # 如果分塊中沒有標題，則使用文檔標題
                        chunked_doc["chunk_title"] = f"{doc['title']} (Part {i+1})"
                    else:
                        chunked_doc["chunk_title"] = f"Section {i+1}"
            # 為 JSON 文件設置標題
            elif doc_type in ["json", "jsonl"]:
                if "title" in chunk:
                    chunked_doc["chunk_title"] = chunk["title"]
                elif "name" in chunk:
                    chunked_doc["chunk_title"] = f"{chunk['name']} (JSON Part {i+1})"
                elif "id" in chunk:
                    chunked_doc["chunk_title"] = f"JSON Item {chunk['id']}"
                else:
                    chunked_doc["chunk_title"] = f"JSON Part {i+1}"
            
            chunked_docs.append(chunked_doc)
    
    logging.info(f"將 {len(documents)} 個文檔分塊為 {len(chunked_docs)} 個區塊")
    return chunked_docs

def preprocess_web_action_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    預處理網頁操作記錄 JSON 格式，過濾不重要的數據
    專門針對 web_actions_*.json 格式設計
    """
    if not json_data or not isinstance(json_data, dict):
        return json_data
    
    # 創建一個新的預處理後的 JSON 對象，保留主要結構
    processed_data = {
        "task_description": json_data.get("task_description", ""),
        "timestamp": json_data.get("timestamp", ""),
        "actions": []
    }
    
    # 如果沒有操作，直接返回
    if "actions" not in json_data or not json_data["actions"]:
        return processed_data
    
    # 過濾並簡化操作序列
    actions = json_data["actions"]
    filtered_actions = []
    
    # 用於追蹤已訪問的頁面，避免重複
    visited_urls = set()
    previous_action_type = None
    
    for action in actions:
        # 基本動作類型
        action_type = action.get("type", "")
        
        # 創建一個簡化版的操作記錄
        simplified_action = {
            "order": action.get("order", 0),
            "type": action_type,
        }
        
        # 根據操作類型進行特定處理
        if action_type == "Navigate":
            url = action.get("url", "")
            # 如果是重複訪問相同的 URL，跳過
            if url in visited_urls:
                continue
            visited_urls.add(url)
            
            simplified_action["url"] = url
            simplified_action["page_title"] = action.get("page_title", "")
            
        elif action_type == "Click":
            # 對於點擊操作，提取元素的關鍵信息
            element = action.get("element", {})
            
            # 只保留有意義的元素屬性
            important_attrs = ["tagName", "text", "id", "className", "name", "type"]
            simplified_element = {attr: element.get(attr, "") for attr in important_attrs if attr in element}
            
            simplified_action["element_info"] = simplified_element
            simplified_action["elements_text"] = action.get("elements_text", "")
            
        elif action_type == "Type":
            # 對於輸入操作，保留輸入的文本
            element = action.get("element", {})
            simplified_action["input_value"] = element.get("value", "")
            simplified_action["elements_text"] = action.get("elements_text", "")
            
        elif action_type == "answer":
            # 保留答案完整信息
            simplified_action["elements_text"] = action.get("elements_text", "")
        
        # 避免連續相同類型的重複操作 (除非是有實質性差異的點擊或輸入)
        if (previous_action_type != action_type or 
            action_type in ["Click", "Type"] or
            action_type == "answer"):
            filtered_actions.append(simplified_action)
            previous_action_type = action_type
    
    processed_data["actions"] = filtered_actions
    return processed_data

def chunk_json_document(doc: Dict[str, Any], text_splitter) -> List[Any]:
    """
    專門處理 JSON 文檔的分塊函數，特別針對 web_actions 類型的 JSON
    預設已經由 preprocess_web_action_json 處理過，專注於操作記錄的結構性分塊
    保留操作的順序性，因為網頁操作的順序非常重要
    """
    chunks = []
    
    # 檢查是否為網頁操作記錄格式（含有 task_description 和 actions）
    if "task_description" in doc and "actions" in doc and isinstance(doc["actions"], list):
        # 策略 1: 將整個預處理後的 JSON 作為一個塊
        # 這適用於相對簡短的操作序列或查詢時需要完整上下文的情況
        if len(doc["actions"]) <= 10:  # 如果操作數量較少，作為一個整體
            chunks.append(doc)
            return chunks
        
        # 策略 2: 對於較長的操作序列，按照順序進行分塊
        # 每個塊包含一系列連續的操作步驟，保持操作的順序關聯
        actions = doc["actions"]
        
        # 每個分塊包含的操作數量
        chunk_size = 5  # 可以根據需要調整
        
        # 按順序分塊
        for i in range(0, len(actions), chunk_size):
            # 獲取當前分塊的操作
            chunk_actions = actions[i:i+chunk_size]
            
            # 創建分塊
            chunk = {
                "task_description": doc["task_description"],
                "timestamp": doc.get("timestamp", ""),
                "sequence_range": f"{i+1}-{min(i+len(chunk_actions), len(actions))}",  # 添加順序範圍標記
                "actions": chunk_actions
            }
            
            # 添加當前分塊的操作類型概述，幫助搜索
            action_types = set(action.get("type", "") for action in chunk_actions)
            chunk["action_types"] = list(action_types)
            
            chunks.append(chunk)
        
        # 如果按順序分塊后沒有產生任何塊，回退到整體處理
        if not chunks:
            chunks.append(doc)
            
        return chunks
    
    # 對於非 web_actions 格式的數據，保持原有處理方式
    if isinstance(doc, dict):
        # JSON 數據通常不適合使用文本分塊器，因為會破壞結構
        # 可以嘗試序列化後分塊，但這裡我們選擇保持結構完整
        chunks.append(doc)
    else:
        # 如果不是字典類型，嘗試轉換為字符串並進行分塊
        try:
            text = json.dumps(doc, ensure_ascii=False)
            chunks = [{"content": chunk} for chunk in text_splitter.split_text(text)]
        except:
            # 如果無法序列化，將其作為一個整體
            chunks.append({"content": str(doc)})
    
    return chunks

def get_retriever_context(task: str, domain: str , webName : str, llm, print_answer: bool = False) -> Optional[str]:
    """實現本地 RAG 功能，從本地知識庫獲取上下文"""
    
    if llm is None:
        return None
    
    try:
        # 先使用 LLM 生成優化的查詢
        optimized_query = generate_optimized_query(task, domain, llm)
        logging.info(f"使用優化後的查詢: {optimized_query}")
        
        print("Start to get retriever context")
        
        # 加載知識庫文檔 - 從 data/markdown 目錄
        #knowledge_base_path = f"data/markdown/{webName}"
        knowledge_base_path = f"data/actionRecoder/{webName}"
        all_documents = load_knowledge_documents(knowledge_base_path)
        
        if not all_documents:
            logging.warning("沒有找到知識庫文檔")
            return None
        
        # 對文檔進行結構化分塊
        chunked_documents = chunk_documents(all_documents)
        
        # 提取文檔內容
        doc_texts = []
        for doc in chunked_documents:
            content = extract_document_content(doc)
            if content:
                doc_texts.append(content)
        
        # 獲取嵌入並執行語義搜索
        if doc_texts:
            embeddings = get_embeddings(doc_texts, llm)
            # 使用優化後的查詢而非原始任務描述
            relevant_docs = semantic_search(optimized_query, chunked_documents, embeddings, llm, top_k=5)
            
            # 如果找到相關文檔，構建上下文內容
            if relevant_docs:
                context_lists = []
                
                # 對檢索到的文檔進行排序和分組
                # 首先按來源文件分組
                docs_by_source = {}
                for doc in relevant_docs:
                    source = doc.get("source", "unknown")
                    if source not in docs_by_source:
                        docs_by_source[source] = []
                    docs_by_source[source].append(doc)
                
                # 然後為每個來源文件的分塊按順序排列
                for source, docs in docs_by_source.items():
                    if len(docs) > 1:
                        # 如果有多個分塊，根據 chunk_id 排序
                        docs.sort(key=lambda x: x.get("chunk_id", 0))
                    
                    # 格式化每個分塊的內容
                    for doc in docs:
                        context = format_document_context(doc)
                        if context:
                            context_lists.append(context)
                


                # Combine context items into a single string with numbered separators
                context_texts = ""
                for i, text in enumerate(context_lists):
                    if text.strip():
                        context_texts += f"\n\n[DOCUMENT {i+1}]\n{text}"

                # 使用 LLM 生成最終答案
                from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

                # 創建系統提示，定義助手的角色和行為
                system_template = f"""You are an assistant designed to help other agents navigate and operate on a webpage. Based on the provided knowledge base, identify relevant information and infer detailed steps necessary to complete the operation. If no relevant information exists in the knowledge base, respond with "No data available" to prevent misleading the original agent's decision-making process.

---

# Knowledge Base Usage Guidelines
1. **Based on the Knowledge Base**: All responses must be grounded in the information provided in the knowledge base. If multiple relevant entries exist, select the most contextually appropriate one for your reply and reasoning.
2. **No Relevant Information**: If the knowledge base lacks the data to answer the query, reply directly with "No data available."
3. **Avoid Speculations**: Do not speculate about scenarios or details not mentioned in the knowledge base.
4. **Step-by-Step Reasoning**: Clearly outline all necessary steps to complete the operation based on the knowledge base and list these steps in a logical sequence.

---

# Response Format

Present responses clearly using either paragraph or bullet point format, with the following sections:
1. **Relevant Knowledge Base Summary**: Include cited content and its contextual relevance.
2. **Operational Logic Reasoning**: Combine the knowledge base content to explain the overall thought process involved in the operation.
3. **Operational Steps**: List concrete and concise steps in sequential order.

If there is no relevant content in the knowledge base, directly respond with:
```plaintext
No data available
```

---

# Example

### Scenario 1: Relevant Information Exists
**User Query**: How do I reset my password on the website?

**Knowledge Base Content**:
- [Password Reset] The website provides a password reset feature. Users must visit the "Forgot Password" page, enter their email address, and click submit.
- [Email Verification] After submission, the system will send a verification email containing a link to reset the password.
- [New Password Requirements] The password must be at least eight characters long and include one number and one special character.

**Response**:
1. **Relevant Knowledge Base Summary**:
   - Password reset requires visiting the "Forgot Password" page.
   - Specific steps involve entering an email, verifying through a system-sent email, and setting a new password adhering to certain requirements.

2. **Operational Logic Reasoning**:
   - Resetting the password involves three stages: email submission, email verification, and password creation. Each stage depends on the completion of the previous one.

3. **Operational Steps**:
   - Navigate to the "Forgot Password" page on the website.
   - Enter your email address and click the "Submit" button.
   - Check your email inbox for the password reset email sent by the system.
   - Click the password reset link provided in the email.
   - Create a new password meeting the following criteria: at least eight characters, includes one number, and one special character.
   - Submit the new password to complete the process.

---

### Scenario 2: No Relevant Information
**User Query**: How do I disable two-factor authentication?

**Knowledge Base Content**: None.

**Response**:
```plaintext
No data available
```

---

# Notes
1. **No Assumptions**: All responses must strictly adhere to the knowledge base, and no ambiguous recommendations should be provided if data is unavailable.
2. **Clarity of Steps**: Steps should be concise yet comprehensive enough to cover the complete process.
3. **Response Consistency**: Maintain consistent formatting and avoid introducing unnecessary ambiguity.
The following is the knowledge base： {context_texts} The above is the knowledge base."""
                
                
                user_prompt = f"""I need to complete the task of '{task}' on the '{domain}' website. What are the most likely methods to achieve this? \n"""

                
                # 使用 OpenAI API 生成回應
                messages = [
                    {"role": "system", "content": system_template},
                    {"role": "user", "content": user_prompt}
                ]
                
                answer = llm.invoke(messages)

                if print_answer and answer:
                    print("\n" + "=" * 50)
                    print("Response:", answer.content)
                
                print("Finish to get retriever context")
                return answer.content
        
        logging.warning("無法找到相關上下文或生成回應") 
        return None
        
    except Exception as e:
        logging.error(f"get_retriever_context 發生錯誤: {str(e)}")
        return None