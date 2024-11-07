import os
import shutil

import gradio as gr

from chat_old import chat, update_person
from read import read, refresh


# 定义更新聊天记录的函数
def update_chat(messages, user_input):
    # 调用 chat 获取回复
    reply = chat(user_input)
    # 将用户输入和回复组成新的对话列表
    messages.append([user_input, reply])
    return messages, ""


# tag1的输入、输出，以及对应处理函数
with gr.Blocks() as app1:

    # 添加输入框，带有默认提示文本
    person_input = gr.Textbox(
        label="初始化人设",
        placeholder="默认设定：现在我们开始一个角色扮演游戏，以下是你的人设：你是尼克·王尔德（Nick Wilde），男，是2016年迪士尼动画电影《疯狂动物城》中的男主角。原型是赤狐。原名尼古拉斯·皮比里厄斯·王尔德（Nicholas Piberius Wilde），在中国大陆地区又被称作狐尼克，由杰森·贝特曼和凯特·索西配音。你少时因遭遇挫折和他人的偏见，被迫放弃了自己的理想，打算不再为谁付出，长大后以坑蒙拐骗为生。你口若悬河、思维敏捷、谎技高超但却内心善良，同时拥有过目不忘的惊人记忆力。因为意外而和动物城警官朱迪·霍普斯被卷进了一个意欲颠覆动物城的巨大阴谋，在案件的侦破过程中，你的超常记忆力和对动物城了如指掌起到了至关重要的作用。案件成功告破后，你通过训练并加入了动物城警察局，成为了一名真正的警察，实现了自己的梦想，正式成为了朱迪的搭档。",
    )

    # 提交按钮，用于触发 update_person 函数
    person_submit_btn = gr.Button("Submit")

    # 点击按钮时触发 update_person 函数，无返回值
    person_submit_btn.click(fn=update_person, inputs=[person_input], outputs=[])

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="输入")

    # 提交按钮，用于提交用户消息
    submit_btn = gr.Button("Submit")

    # submit_btn 触发 update_chat 函数，更新聊天记录
    submit_btn.click(fn=update_chat, inputs=[chatbot, msg], outputs=[chatbot, msg])

    # 让 Textbox 的回车行为也触发 submit
    msg.submit(fn=update_chat, inputs=[chatbot, msg], outputs=[chatbot, msg])

    # 自定义清除按钮，清除后触发 update_person
    def clear_and_update_person():
        return (
            "",
            [],
        )  # 清空文本框和聊天记录

    # 使用普通按钮替代 ClearButton，执行自定义清除逻辑
    clear_btn = gr.Button("Clear")

    # 按下清除按钮后，清空内容并调用 update_person 函数
    clear_btn.click(fn=clear_and_update_person, inputs=[], outputs=[msg, chatbot])
    clear_btn.click(fn=update_person, inputs=[person_input], outputs=[])


# 定义保存文件的函数
def save_file(file):
    # 将上传的文件保存为 temp.txt
    save_path = os.path.join(os.getcwd(), "temp.txt")
    shutil.move(file.name, save_path)
    return refresh()


# tag2的输入、输出，以及对应处理函数
# app2 = gr.Interface(
#     fn=read,
#     inputs=gr.Textbox(label="向文档提问"),
#     outputs=gr.Textbox(label="结果"),
# )
with gr.Blocks() as app2:
    upload_interface = gr.Interface(fn=save_file, inputs="file", outputs="text")

    # 原有的文本输入和输出
    read_input = gr.Textbox(label="向文档提问")
    read_output = gr.Textbox(label="结果")
    # 读取文档的处理函数
    read_interface = gr.Interface(fn=read, inputs=read_input, outputs=read_output)


demo = gr.TabbedInterface(
    [app1, app2],
    tab_names=["角色扮演聊天", "长文档阅读"],
    title="大作业",
)

demo.launch(server_name="0.0.0.0")
