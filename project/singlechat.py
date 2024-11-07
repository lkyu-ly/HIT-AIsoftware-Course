import requests
import json
from time import sleep


def get_access_token():
    # 使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key

    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=FmzJk7hmF6oF7e3K6yCRapkY&client_secret=tyoIldmLMrkCVyvTDKFtnrM4t4KEqHpq"
    # API："https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=[应用API Key]&client_secret=[应用Secret Key]"

    payload = json.dumps("")  # 空字符串转换为JSON格式
    headers = {
        "Content-Type": "application/json",  # 请求体的媒体类型（在这里是JSON）
        "Accept": "application/json",  # 客户端期望接收的数据类型（同样是JSON）
    }

    response = requests.request(
        "POST", url, headers=headers, data=payload
    )  # 使用requests库的request方法发送一个POST请求到指定的url。请求中包含之前定义的头部信息headers和数据体payload
    return response.json().get(
        "access_token"
    )  # 使用response.json()方法将响应体解析为JSON格式，然后调用get方法获取键名为access_token的值


def model(url, user_input, system_1, matrix, n):

    # 记忆信息处理
    mem = ""
    if len(matrix) <= n:
        for m in matrix:
            mem += m
    else:
        for m in matrix[-n:]:
            mem += m

    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": user_input}],
            "stream": False,  # 是否以流式接口的形式返回数据
            "temperature": 0.9,  # 输出的随机性
            "top_p": 0.7,  # 输出文本的多样性
            "penalty_score": 1,  # 已生成的token增加惩罚
            "system": system_1
            + "你现在在与用户和你的搭档朱迪/尼克一共两人对话，其中第一句是用户，第二句是你的搭档"
            + "之前对话信息"
            + mem,  # 模型人设
            "max_output_tokens": 4096,  # 最大输出token数
            "frequency_penalty": 0.1,  # 迄今为止文本中的现有频率对新token进行惩罚，降低重复性
            "presence_penalty": 0.0,  # token记目前是否出现在文本中来对其进行惩罚，提高创新性
        }
    )  # 发送的数据

    headers = {"Content-Type": "application/json"}  # 请求体的内容类型为JSON

    response_1 = requests.request("POST", url, headers=headers, data=payload)

    try:
        response_1_data = response_1.json()  # 响应体解析为Python字典

        if "result" in response_1_data:
            print(response_1_data["result"])
            # print("\n")
        else:
            print("没有返回结果或结果格式不正确。")
    except json.JSONDecodeError:
        print("无法解析JSON响应。")

    # 使用大模型提取关键字作为记忆
    payload = json.dumps(
        {
            "messages": [
                {"role": "user", "content": user_input + response_1_data["result"]}
            ],
            "stream": False,  # 是否以流式接口的形式返回数据
            "temperature": 0.9,  # 输出的随机性
            "top_p": 0.7,  # 输出文本的多样性
            "penalty_score": 1,  # 已生成的token增加惩罚
            "system": "提取简洁关键词",  # 模型人设
            "max_output_tokens": 4096,  # 最大输出token数
            "frequency_penalty": 0.1,  # 迄今为止文本中的现有频率对新token进行惩罚，降低重复性
            "presence_penalty": 0.0,  # token记目前是否出现在文本中来对其进行惩罚，提高创新性
        }
    )  # 发送的数据

    headers = {"Content-Type": "application/json"}  # 请求体的内容类型为JSON

    response_2 = requests.request("POST", url, headers=headers, data=payload)
    response_2_data = response_2.json()
    # print(response_2_data['result'])

    return response_1_data["result"], response_2_data["result"]


url = (
    "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token="
    + get_access_token()
)

n = 3  # 记忆的最长对话轮数
system_1 = "我们现在进行角色扮演，这是你的人设：狐狸尼克是一只机智、狡黠却内心善良的狐狸。他拥有超凡的记忆力，对动物城了如指掌。曾以坑蒙拐骗为生，但在与朱迪警官共同侦破案件后，他找到了新的人生方向，成为了一名真正的警察，实现了自己的梦想。"  # 对话对象，可自行设定
matrix_1 = []  # 对话信息记录
answer_1 = ""

system_2 = "我们现在进行角色扮演，这是你的人设：兔子朱迪是是一只立志成为警察的兔子，通过自己的奋斗，成为了动物城的第一个兔子警官。尽管面临来自同僚的傲慢与偏见，朱迪凭借过人的胆识和智慧，最终侦破了大规模失踪案，并与‌狐狸尼克成为搭档，揭示了案件背后的阴谋。她的故事激励了无数人，展示了弱小外表下坚韧不拔的灵魂和永不放弃的梦想。"
matrix_2 = []  # 对话信息记录
answer_2 = ""

while True:
    user_input = input("请输入您的问题（输入'quit'退出）：")
    if user_input.lower() == "quit":
        break
    answer_1, data_1 = model(url, answer_2 + "\n" + user_input, system_1, matrix_1, n)
    matrix_1.append(data_1[:])
    answer_2, data_2 = model(url, "\n" + answer_1 + user_input, system_2, matrix_2, n)
    matrix_2.append(answer_2[:])
