import platform
import argparse
import time
import json
import re
import os
import shutil
import logging
from typing import Annotated, Literal, Dict, Any

from typing_extensions import TypedDict
from openai import RateLimitError, APIError
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from io import BytesIO

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureOpenAI,OpenAI,AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service as ChromeService

from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY,SYSTEM_PROMPT_TYPE
from utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only

from RagFlow import RagflowAPIConfig , RagflowAPI 

class State(TypedDict):
    task: Annotated[str, "The task to be completed"]
    auto_messages: Annotated[list, add_messages]
    messages: Annotated[list, "messages"]
    driver: Annotated[object, "Selenium WebDriver instance"]
    download_files: Annotated[list, "List of downloaded files"]
    iteration: Annotated[int, "Current iteration count"]
    args: Annotated[dict, "Program arguments"]
    task_dir: Annotated[str, "Task directory path"]
    web_elements: Annotated[dict, "Current page web elements"]
    fail_obs: Annotated[str, "Failure observation"]
    pdf_obs: Annotated[str, "PDF observation"]
    warn_obs: Annotated[str, "Warning observation"]
    llm : Annotated[object, "OpenAI API instance"]
    current_response : Annotated[str, "Current response from the assistant"]
    current_screenshot : Annotated[str, "Current screenshot"]
    LLM_Cost : Annotated[float, "Total cost of LLM API"]
    RetrieverContext : Annotated[str, "Retriever Context"]

def driver_config(args):
    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )
    return options

def setup_environment(args):
    current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    result_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def GetRetrieverContext(ragConfig ,Task , Domain , print_answer = False) -> Dict[str, Any]:
    
    if ragConfig.base_url == "" or ragConfig.chat_id == "" or ragConfig.api_key == "":
        #return "There are no retriever context information"
        return None

    #task = "查詢資訊工程學系碩士班的課程中，訊息理解與Web智慧這門課的授課教授是誰?"
    #domain = "https://cis.ncu.edu.tw/Course/main/news/announce"
    query = f"I need to complete the task of '{Task}' on the '{Domain}' website. What are the most likely methods to achieve this?"

    api = RagflowAPI(ragConfig)

    session_id = api.create_session()
    if not session_id:
        logging.error("Failed to create session")
        return

    print("Start to get retriever context")
    answer = api.get_completion(query, session_id)

    if print_answer:
        if answer:
            print("\n" + "=" * 50)
            print("Response:", answer)
        else:
            print("Failed to get response")

    print("Finish to get retriever context")
    return answer

def launchBrowser(state: State):
    args = state["args"]
    task = state["task"]
    #task_dir = state["task_dir"]
    
    # 根據參數設定配置Chrome瀏覽器選項
    options = driver_config(args)
    
    # 初始化Chrome瀏覽器驅動
    driver = webdriver.Chrome(options=options)
    # 設置瀏覽器視窗大小
    driver.set_window_size(args.window_width, args.window_height)
    # 導航到任務指定的網頁
    driver.get(task['web'])
    
    try:
        # 等待頁面載入完成
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support.expected_conditions import staleness_of
        
        WebDriverWait(driver, 10).until(
            lambda driver: driver.execute_script('return document.readyState') == 'complete'
        )
    except:
        pass
    
    # 防止空白鍵的預設行為(頁面滾動)，除非焦點在文本輸入框
    driver.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
    
    # 更新狀態：設置驅動器、初始化下載文件列表和迭代計數器
    state["driver"] = driver
    state["download_files"] = []
    state["iteration"] = 0

    # 初始化RagFlow API配置
    ragFlowConfig = RagflowAPIConfig(
        base_url = args.ragFlow_url,
        chat_id = args.ragFlow_chat_id,
        api_key = args.ragFlow_api_key
    )

    state["RetrieverContext"] = GetRetrieverContext(ragFlowConfig,task['ques'], task['web'])
    #state["RetrieverContext"] = None
    return state

def format_observation(state: State):
    state["iteration"] += 1
    if not state["fail_obs"]:
        driver = state["driver"]
        args = state["args"]
        
        if not args.text_only:
            rects, web_eles, web_eles_text = get_web_element_rect(driver, fix_color=args.fix_box_color)
            state["web_elements"] = {
                "rects": rects,
                "elements": web_eles,
                "text": web_eles_text
            }
        else:
            accessibility_tree_path = os.path.join(state["task_dir"], f'accessibility_tree{state["iteration"]}')
            ac_tree, obs_info = get_webarena_accessibility_tree(driver, accessibility_tree_path)
            state["web_elements"] = {
                "ac_tree": ac_tree,
                "obs_info": obs_info
            }
            
        img_path = os.path.join(state["task_dir"], f'screenshot{state["iteration"]}.png')
        driver.save_screenshot(img_path)
        
        state["current_screenshot"] = encode_image(img_path)

        return state
    
    return state
def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text, retriever_context):
    if it == 1:
        #init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that a <textarea> or <input> may be a textbox, but not exactly. Not all elements are in the screenshot. You can identify them by visible or invisible words. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ],
            'context': retriever_context
        }
        init_msg_format['content'].append({"type": "image_url",
                                           "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Not all elements are in the screenshot. You can identify them by visible or invisible words. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ],
                'context': retriever_context
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Not all elements are in the screenshot. You can identify them by visible or invisible words. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ],
                'context': retriever_context
            }
        return curr_msg


def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        init_msg_format = {
            'role': 'user',
            'content': init_msg + '\n' + ac_tree
        }
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
            }
        return curr_msg

def thoughts(state: State):
    args = state["args"]

    # 僅在 state["messages"] 為空時加入 system message (避免重置先前訊息)
    if not state["messages"]:
        if args.text_only:
            state["messages"].append({'role': 'system', 'content': SYSTEM_PROMPT_TEXT_ONLY})
        else:
            state["messages"].append({'role': 'system', 'content': SYSTEM_PROMPT_TYPE})
    
    obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
    if args.text_only:
        obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."
    
    # ...existing code...
    init_msg = f"""Now given a task: {state['task']['ques']}  Please interact with https://www.example.com and get the answer. \n"""
    init_msg = init_msg.replace('https://www.example.com', state['task']['web'])
    

    # state["RetrieverContext"] is not null and no contain "No data available."
    if (state["RetrieverContext"] and "No data available." not in state["RetrieverContext"]):
        init_msg = init_msg + "Here's the following operating manual provides suggestions: " + state["RetrieverContext"]
    # clear state["RetrieverContext"] 
    state["RetrieverContext"] = None
    
    init_msg = init_msg + obs_prompt

    if not args.text_only:
        curr_msg = format_msg(
            state["iteration"], 
            init_msg,
            state["pdf_obs"],
            state["warn_obs"],
            state["current_screenshot"],
            state["web_elements"]["text"],
            ''
        )
    else:
        curr_msg = format_msg_text_only(
            state["iteration"],
            init_msg,
            state["pdf_obs"],
            state["warn_obs"], 
            state["web_elements"]["ac_tree"]
        )
        
    state["messages"].append(curr_msg)
    
    # Clip messages, too many attached images may cause confusion
    if not args.text_only:
        state["messages"] = clip_message_and_obs(state["messages"], args.max_attached_imgs)
    else:
        state["messages"] = clip_message_and_obs_text_only(state["messages"], args.max_attached_imgs)

    # Call GPT-4V API and process response
    prompt_tokens, completion_tokens, gpt_call_error, openai_response = call_gpt4v_api(args, state["llm"], state["messages"])
    
    if gpt_call_error:
        # Add error handling for token counting
        logging.error('API call failed')
        state["fail_obs"] = "OpenAI API call failed. Please try again."
        return state
    
    state["LLM_Cost"]["accumulate_prompt_token"] += prompt_tokens
    state["LLM_Cost"]["accumulate_completion_token"] += completion_tokens
    logging.info(f'Accumulate Prompt Tokens: {state["LLM_Cost"]["accumulate_prompt_token"]}; Accumulate Completion Tokens: {state["LLM_Cost"]["accumulate_completion_token"]}')
    logging.info('API call complete...')
        
    #gpt_response = openai_response.choices[0].message.content
    gpt_response = openai_response.content
    state["messages"].append({'role': 'assistant', 'content': gpt_response})
    state["current_response"] = gpt_response
    
    

    return state

# 首先新增這些輔助函數
def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)

def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    
    try:
        web_ele.clear()
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)
    #actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs

def exec_action_scroll(info, web_eles, driver_task, args, obs_info):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {args.window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-args.window_height*2//3});")
    else:
        if not args.text_only:
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            element_box = obs_info[scroll_ele_number]['union_bound']
            element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)

# 修改 action 函數
def action(state: State):
    response = state["current_response"]
    chosen_action = re.split(r'Thought:|Action:|Observation:', response)[2].strip()
    action_key, info = extract_information(chosen_action)
    
    # 如果是 answer action，直接返回不再循環
    if action_key == 'answer':
        logging.info(f'Final Answer: {info["content"]}')
        with open(os.path.join(state["task_dir"], "answer.txt"), "w", encoding='utf-8' ) as f:
            f.write(info["content"])
        return state

    driver = state["driver"]
    args = state["args"]
    web_elements = state["web_elements"]
    state["fail_obs"] = ""
    state["pdf_obs"] = ""
    state["warn_obs"] = ""
    
    try:
        window_handle = driver.current_window_handle
        driver.switch_to.window(window_handle)

        if action_key == 'click':
            if not args.text_only:
                click_ele_number = int(info[0])
                web_ele = web_elements["elements"][click_ele_number]
            else:
                click_ele_number = info[0]
                element_box = web_elements["obs_info"][click_ele_number]['union_bound']
                element_box_center = (element_box[0] + element_box[2] // 2,
                                    element_box[1] + element_box[3] // 2)
                web_ele = driver.execute_script(
                    "return document.elementFromPoint(arguments[0], arguments[1]);",
                    element_box_center[0], element_box_center[1]
                )
            
            exec_action_click(info, web_ele, driver)
            
            # Handle PDF download
            current_files = sorted(os.listdir(args.download_dir))
            if current_files != state["download_files"]:
                time.sleep(10)
                current_files = sorted(os.listdir(args.download_dir))
                
                current_download_file = [
                    pdf_file for pdf_file in current_files 
                    if pdf_file not in state["download_files"] and pdf_file.endswith('.pdf')
                ]
                
                if current_download_file:
                    pdf_file = current_download_file[0]
                    state["pdf_obs"] = get_pdf_retrieval_ans_from_assistant(
                        state["llm"], 
                        os.path.join(args.download_dir, pdf_file), 
                        state["task"]['ques']
                    )
                    shutil.copy(
                        os.path.join(args.download_dir, pdf_file),
                        state["task_dir"]
                    )
                    state["pdf_obs"] = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + state["pdf_obs"]
                
                state["download_files"] = current_files

        elif action_key == 'type':
            if not args.text_only:
                type_ele_number = int(info['number'])
                web_ele = web_elements["elements"][type_ele_number]
            else:
                type_ele_number = info['number']
                element_box = web_elements["obs_info"][type_ele_number]['union_bound']
                element_box_center = (element_box[0] + element_box[2] // 2,
                                    element_box[1] + element_box[3] // 2)
                web_ele = driver.execute_script(
                    "return document.elementFromPoint(arguments[0], arguments[1]);",
                    element_box_center[0], element_box_center[1]
                )
            
            state["warn_obs"] = exec_action_type(info, web_ele, driver)

        elif action_key == 'scroll':
            if not args.text_only:
                exec_action_scroll(info, web_elements["elements"], driver, args, None)
            else:
                exec_action_scroll(info, None, driver, args, web_elements["obs_info"])

        elif action_key == 'wait':
            time.sleep(5)

        elif action_key == 'goback':
            driver.back()
            time.sleep(2)

        elif action_key == 'google':
            driver.get('https://www.google.com/')
            time.sleep(2)

        elif action_key == 'answer':
            logging.info(info['content'])
            logging.info('finish!!')
            return state

        else:
            raise NotImplementedError

    except Exception as e:
        logging.error('Driver error info:')
        logging.error(e)
        if 'element click intercepted' not in str(e):
            state["fail_obs"] = "The action you have chosen cannot be executed. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
        time.sleep(2)
    
    
    return state

def has_answer(state: State) -> Literal["action", "answer"]:
    
    #if not state["current_response"]:
    #    return "action"
    
    # 檢查迭代次數
    if state["iteration"] >= state["args"].max_iter:
        logging.info("Reached maximum iterations, forcing answer...")
        state["current_response"] = "Thought: Maximum iterations reached.\nAction: ANSWER; 'Task could not be completed within the maximum allowed iterations.'"
        return "answer"

    response = state["current_response"]
    try:
        chosen_action = re.split(r'Thought:|Action:|Observation:', response)[2].strip()
        action_key, _ = extract_information(chosen_action)
        return "answer" if action_key == "answer" else "action"
    except:
        return "action"

def answer(state: State):
    response = state["current_response"]
    answer_content = re.split(r'Thought:|Action:|Observation:', response)[2].strip()
    
    with open(os.path.join(state["task_dir"], "answer.txt"), "w", encoding='utf-8') as f:
        f.write(answer_content)
        f.close()
    
    print_message(state["messages"], state["task_dir"])
    state["driver"].quit()
    logging.info(f'Total cost: {state["LLM_Cost"]["accumulate_prompt_token"] / 1000 * 0.01 + state["LLM_Cost"]["accumulate_completion_token"] / 1000 * 0.03}')


    return state

def showImage(image):
    pil_image = PILImage.open(BytesIO(image))
    plt.imshow(pil_image)
    plt.axis('off')  # Hide axes
    plt.show()  # Display the image

def call_gpt4v_api(args, llm, messages):
    retry_times = 0
    while True:
        try:
            if not args.text_only:
                logging.info('Calling gpt4v API...')
                response = llm.invoke(messages)
            else:
                logging.info('Calling gpt4 API...')
                response = llm.invoke(messages, timeout=30)

            # Extract token usage from response metadata
            #prompt_tokens = response.metadata.get('prompt_tokens', 0)
            #completion_tokens = response.metadata.get('completion_tokens', 0)
            token_usage = response.response_metadata.get('token_usage', {})
            prompt_tokens = token_usage.get('prompt_tokens', 0)
            completion_tokens = token_usage.get('completion_tokens', 0)

            logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

            return prompt_tokens, completion_tokens, False, response

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')

            if isinstance(e, RateLimitError):
                time.sleep(10)
            elif isinstance(e, APIError):
                time.sleep(15)
            else:
                return None, None, True, None

            retry_times += 1
            if retry_times == 10:
                logging.info('Retrying too many times')
                return None, None, True, None

def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument("--api_key", default="key", type=str, help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="gpt-4-vision-preview", type=str, help="api model name")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true')
    # for web browser
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)  # for headless mode, there is no address bar
    parser.add_argument("--fix_box_color", action='store_true')
    parser.add_argument("--azure_endpoint", type=str, default="")
    parser.add_argument("--api_version", type=str, default="")
    parser.add_argument("--ragFlow_url", type=str, default="")
    parser.add_argument("--ragFlow_chat_id", type=str, default="")
    parser.add_argument("--ragFlow_api_key", type=str, default="")


    args = parser.parse_args()

    #options = driver_config(args)

    llm = AzureChatOpenAI(
        api_key=args.api_key,
        model=args.api_model,
        api_version=args.api_version,
        temperature=args.temperature,
        azure_endpoint=args.azure_endpoint
    )

    # Save Result file
    result_dir = setup_environment(args)

    # Load tasks
    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    #prompt = ChatPromptTemplate.from_template("prompt_str")
    #chain = prompt | llm

    # Initialize graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("launchBrowser", launchBrowser)
    workflow.add_node("observation", format_observation)
    workflow.add_node("thoughts", thoughts)
    workflow.add_node("action", action)
    workflow.add_node("answer", answer)
    
    # Add edges
    workflow.add_edge(START, "launchBrowser")
    workflow.add_edge("launchBrowser", "observation")
    workflow.add_edge("observation", "thoughts")
    workflow.add_conditional_edges(
        "thoughts",
        has_answer,
        {
            "answer": "answer",
            "action": "action"
        }
    )
    workflow.add_edge("action", "observation")  # action 完成後回到 observation
    workflow.add_edge("answer", END)
    
    # Compile and run
    graph = workflow.compile()
    
    # Load tasks and execute
    for task in tasks:

        task_dir = os.path.join(result_dir, f'task{task["id"]}')
        os.makedirs(task_dir, exist_ok=True)
        setup_logger(task_dir)
        logging.info(f'########## TASK{task["id"]} ##########')
        print(f'########## TASK{task["id"]} ##########')

        cost = {
            "accumulate_prompt_token": 0,
            "accumulate_completion_token": 0
        }

        initial_state = {
            "task": task,
            "args": args,
            "messages": [],
            "task_dir": task_dir,
            "llm": llm,
            "fail_obs": "",
            "pdf_obs": "",
            "warn_obs": "",
            "web_elements": {},
            "download_files": [],
            "iteration": 0,
            "driver": None,
            "current_response": None,
            "LLM_Cost": cost
        }
        
        try:
            result = graph.invoke(initial_state, {"recursion_limit": 100})
            logging.info(f"Task {task['id']} completed successfully")
        except Exception as e:
            import traceback
            logging.error(f"Task {task['id']} failed: {str(e)}")
            traceback.print_exc()
            continue

    #image = graph.get_graph().draw_mermaid_png()
    #showImage(image)
    
if __name__ == '__main__':
    main()