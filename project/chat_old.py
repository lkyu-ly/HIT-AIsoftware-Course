from calendar import c
import json

import requests

# 用于存储对话的历史记录
conversation_history = []
person = {
    "messages": conversation_history,  # 发送对话历史记录
    "stream": False,
    "temperature": 0.9,
    "top_p": 0.7,
    "penalty_score": 1,
    "system": "",
    "max_output_tokens": 4096,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.0,
}


def get_access_token():
    # 使用 API Key，Secret Key 获取access_token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=114514&client_secret=1919810"

    response = requests.post(url)
    return response.json().get("access_token")  # 获取 access_token


def send_request(user_input):

    url = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token="
        + get_access_token()
    )

    headers = {"Content-Type": "application/json"}
    payload = json.dumps(person)
    response = requests.post(url, headers=headers, data=payload)
    response_data = response.json()

    if "result" in response_data:
        # 添加AI的回复到对话历史中
        conversation_history.append(
            {"role": "assistant", "content": response_data["result"]}
        )
        return response_data["result"]
    else:
        return "没有返回结果或结果格式不正确。"


def update_person(text):
    conversation_history = []
    global person

    if text == "":
        person["system"] = (
            "现在我们开始一个角色扮演游戏，以下是你的人设：你是尼克·王尔德（Nick Wilde），男，是2016年迪士尼动画电影《疯狂动物城》中的男主角。原型是赤狐。原名尼古拉斯·皮比里厄斯·王尔德（Nicholas Piberius Wilde），在中国大陆地区又被称作狐尼克，由杰森·贝特曼和凯特·索西配音。你少时因遭遇挫折和他人的偏见，被迫放弃了自己的理想，打算不再为谁付出，长大后以坑蒙拐骗为生。你口若悬河、思维敏捷、谎技高超但却内心善良，同时拥有过目不忘的惊人记忆力。因为意外而和动物城警官朱迪·霍普斯被卷进了一个意欲颠覆动物城的巨大阴谋，在案件的侦破过程中，你的超常记忆力和对动物城了如指掌起到了至关重要的作用。案件成功告破后，你通过训练并加入了动物城警察局，成为了一名真正的警察，实现了自己的梦想，正式成为了朱迪的搭档。"
        )
    else:
        person["system"] = text


def chat(user_input):
    if person["system"] == "":
        return "请先初始化人设"
    else:
        # 添加用户输入到对话历史中
        conversation_history.append({"role": "user", "content": user_input})
        person["messages"] = conversation_history
        print(conversation_history)
        return send_request(user_input)


if __name__ == "__main__":
    chat()
